# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

"""
Jiles-Atherton solver.
"""

from __future__ import annotations
from typing import Any
from logging import getLogger
import dataclasses
import numpy as np
import numpy.typing as npt
from .util import njit
from .mag import HysteresisLoop


class SolverError(RuntimeError):
    pass


class ConvergenceError(SolverError):
    pass


class NumericalError(SolverError):
    pass


@dataclasses.dataclass(frozen=True)
class Coef:
    """
    Per the original model definition, all of them can be scalars or 3x3 matrices:

        Symbol  Description                                                         Range           Unit
        c_r     magnetization reversibility (1 for purely anhysteretic material)    [0, 1]          dimensionless
        M_s     saturation magnetization                                            non-negative    ampere/meter
        a       domain wall density                                                 non-negative    ampere/meter
        k_p     pinning loss                                                        non-negative    ampere/meter
        alpha   interdomain coupling                                                non-negative    dimensionless
    """

    c_r: float
    M_s: float
    a: float
    k_p: float
    alpha: float

    def __post_init__(self) -> None:
        def non_negative(x: float) -> bool:
            return x >= 0 and np.isfinite(x)

        if not 0 <= self.c_r <= 1:
            raise ValueError(f"c_r invalid: {self.c_r}")
        if not non_negative(self.M_s):
            raise ValueError(f"M_s invalid: {self.M_s}")
        if not non_negative(self.a):
            raise ValueError(f"a invalid: {self.a}")
        if not non_negative(self.k_p):
            raise ValueError(f"k_p invalid: {self.k_p}")
        if not non_negative(self.alpha):
            raise ValueError(f"alpha invalid: {self.alpha}")


COEF_COMSOL_JA_MATERIAL = Coef(c_r=0.1, M_s=1.6e6, a=560, k_p=1200, alpha=0.0007)
"""
Values are taken from the material properties named "Jiles-Atherton Hysteretic Material" in COMSOL Multiphysics.
Useful for testing and validation.
"""


@dataclasses.dataclass(frozen=True)
class Solution:
    """
    Each curve segment contains an nx2 matrix, where the columns are the applied field H and magnetization M.
    """

    virgin: npt.NDArray[np.float64]
    """
    The virgin curve traced from H=0, M=0 to saturation in the positive direction.
    Each row contains the H and M values.
    """

    loop: HysteresisLoop


def solve(
    coef: Coef,
    *,
    tolerance: float = 1e-3,
    saturation_susceptibility: float = 0.05,
    H_stop_range: tuple[float, float] = (100e3, 3e6),
    max_iter: int = 10**9,
    no_balancing: bool = False,
) -> Solution:
    """
    Solves the JA model for the given coefficients and initial conditions by sweeping the magnetization in the
    specified direction using the specified step size until saturation, and then retraces the magnetization back
    until saturation in the opposite direction.

    At the moment, only scalar definition is supported, but this may be extended if necessary.

    The sweeps will stop when either the magnetic susceptibility drops below the specified threshold (which indicates
    that the material is saturated) or when the applied field magnitude exceeds ```H_stop_range[1]```.
    The saturation will not be detected unless the H-field magnitude is at least ```H_stop_range[0]```;
    this is done to handle materials that exhibit very low susceptibility at weak fields.

    The solver will stop the sweep early if a floating point error is raised.
    This can be paired with ``np.seterr(over="raise")`` to simply utilize the maximum range of the floating point
    representation without any special handling.

    If the sweep requires more than ``max_iter`` points, a SolverError will be raised.
    In that case, either the ``max_iter`` needs to be raised, or the tolerance needs to be relaxed.
    """
    c_r, M_s, a, k_p, alpha = coef.c_r, coef.M_s, coef.a, coef.k_p, coef.alpha

    def sweep(H0: float, M0: float, sign: int) -> npt.NDArray[np.float64]:
        assert sign in (-1, +1)
        # This is allocated from the virtual memory anyway, so we can afford to be generous.
        # No mapping to the physical memory is done until we actually attempt to write to the array.
        hm = np.empty((max_iter, 2), dtype=np.float64)
        hm[0] = H0, M0  # The initial point shall be set by the caller.
        idx = np.array([0], dtype=np.uint32)
        try:
            status = _solve(
                hm_out=hm,
                idx_out=idx,
                c_r=c_r,
                M_s=M_s,
                a=a,
                k_p=k_p,
                alpha=alpha,
                direction=sign,
                tolerance=tolerance,
                saturation_susceptibility=saturation_susceptibility,
                H_stop_range=H_stop_range,
            )
        except (FloatingPointError, ZeroDivisionError) as ex:
            status = "Too few points in the sweep" if idx[0] < 10 else ""
            H, M = hm[idx[0]]
            _logger.warning(
                "Sweep stopped at #%s, H=%+.3f, M=%+.3f with %s due to numerical error: %s", idx[0], H, M, coef, ex
            )
            _logger.debug("Stack trace", exc_info=True)

        hm = hm[: idx[0] + 1]
        hm.setflags(write=False)
        H, M = hm[-1]
        if status:
            arrow = " ↑↓"[sign]
            raise ConvergenceError(f"{arrow}#{idx[0]} H0={H0:+.3f} H={H:+.3f} M0={M0:.3f} M={M:+.3f}: {status}")
        return hm

    hm_virgin = sweep(0, 0, +1)

    H_last, M_last = hm_virgin[-1]
    hm_maj_dsc = sweep(H_last, M_last, -1)

    H_last, M_last = hm_maj_dsc[-1]
    hm_maj_asc = sweep(H_last, M_last, +1)

    loop = HysteresisLoop(descending=hm_maj_dsc[::-1], ascending=hm_maj_asc)
    assert loop.descending[0, 0] < loop.descending[-1, 0]
    assert loop.ascending[0, 0] < loop.ascending[-1, 0]
    if (
        not no_balancing
        and loop.descending[0, 0] < 0 < loop.descending[-1, 0]
        and loop.ascending[0, 0] < 0 < loop.ascending[-1, 0]
    ):
        loop = loop.balance()

    return Solution(virgin=hm_virgin, loop=loop)


@njit(nogil=True)
def _solve(
    hm_out: npt.NDArray[np.float64],
    idx_out: npt.NDArray[np.uint32],
    c_r: float,
    M_s: float,
    a: float,
    k_p: float,
    alpha: float,
    direction: int,
    tolerance: float,
    saturation_susceptibility: float,
    H_stop_range: tuple[float, float],
) -> str:
    """
    This function is very bare-bones because it has to be compilable in the nopython mode.
    We can't handle exceptions, can't log, etc, just numerical stuff; the flip side is that it is super fast.
    The caller needs to jump through some extra hoops to use this function, but it is worth it.
    """
    assert direction in (-1, +1)
    assert len(hm_out.shape) == 2 and (hm_out.shape[1] == 2)
    assert idx_out.shape == (1,)
    assert idx_out[0] == 0

    # Dormand–Prince (RK45) constants for single dimension
    C2, C3, C4, C5 = 1 / 5, 3 / 10, 4 / 5, 8 / 9
    A21 = 1 / 5
    A31, A32 = 3 / 40, 9 / 40
    A41, A42, A43 = 44 / 45, -56 / 15, 32 / 9
    A51, A52, A53, A54 = 19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729
    A61, A62, A63, A64, A65 = 9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656
    # 5th-order weights
    B1, B2, B3, B4, B5, B6 = 35 / 384, 0.0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84
    # 4th-order weights (for the embedded estimate)
    B1_4, B2_4, B3_4, B4_4, B5_4, B6_4 = 5179 / 57600, 7571 / 16695, 393 / 640, -92097 / 339200, 187 / 2100, 1 / 40

    # Entities that are constant for the entire sweep.
    max_iter = hm_out.shape[0]
    df = lambda h, m: _dM_dH(c_r=c_r, M_s=M_s, a=a, k_p=k_p, alpha=alpha, H=h, M=m, direction=direction)

    # Entities that can be changed if the solver fails and has to retry.
    # The initial settings assume a non-stiff equation so that we can get a quick solution.
    target_rel_err = 0.1  # Keep local_error/tolerance close to this value.
    dH_abs_range = np.array([1e-6, 10], dtype=np.float64)
    retry_count = 0

    # Entities that are dynamically adjusted during the sweep.
    dH_abs = 1e-3  # This is just a guess that will be dynamically refined.

    assert max_iter > 1
    assert np.isfinite(hm_out[0]).all()  # We require that the first point is already filled in.
    while idx_out[0] < (max_iter - 1):
        dH = direction * dH_abs
        H, M = hm_out[idx_out[0]]

        # Integrate using a high-order method because the equation can be stiff at high susceptibility.
        # _dM_dH() may throw FloatingPointError or ZeroDivisionError; we can't handle them in the compiled code.
        # Since we always update the current state in the caller's context, we can still return partial solution
        # even if an exception is raised.
        k1 = df(H, M)
        k2 = df(H + C2 * dH, M + dH * (A21 * k1))
        k3 = df(H + C3 * dH, M + dH * (A31 * k1 + A32 * k2))
        k4 = df(H + C4 * dH, M + dH * (A41 * k1 + A42 * k2 + A43 * k3))
        k5 = df(H + C5 * dH, M + dH * (A51 * k1 + A52 * k2 + A53 * k3 + A54 * k4))
        k6 = df(H + dH, M + dH * (A61 * k1 + A62 * k2 + A63 * k3 + A64 * k4 + A65 * k5))

        M5 = M + dH * (B1 * k1 + B2 * k2 + B3 * k3 + B4 * k4 + B5 * k5 + B6 * k6)
        M4 = M + dH * (B1_4 * k1 + B2_4 * k2 + B3_4 * k3 + B4_4 * k4 + B5_4 * k5 + B6_4 * k6)

        err_local = np.abs(M5 - M4)
        H_new = H + dH
        M_new = M5

        # Compute the adjustment to the step size.
        e = err_local / tolerance
        adj = min(max(target_rel_err / (e + 1e-9), 0.1), 1.01)

        # Adjust the step size keeping it within the allowed range.
        dH_abs = min(max(dH_abs * adj, float(dH_abs_range[0])), float(dH_abs_range[1]))

        # If the error still exceeds the requested tolerance, adjust the settings and retry.
        # Note that this equation can be quite stiff depending on the parameters, so merely taking a few steps
        # back is not enough: if the error went out of bounds, that means the divergence started many steps ago.
        if e > 1:
            # Uh-oh, this is a stiff case. Restart in finer steps.
            if retry_count == 0:
                retry_count += 1
                target_rel_err *= 0.01
                dH_abs_range *= 0.01
                idx_out[0] = 0  # Nuke everything and begin again.
                continue
            # If we're already at the limit, accept a very large error and hope for the best.
            if e > 100:
                return "Error too large"

        # Save the next point. This ensures that the caller will get partial results even if an exception is raised.
        idx = idx_out[0] + 1
        idx_out[0] = idx
        hm_out[idx] = H_new, M_new

        # Termination check and logging. No need to log the termination reason because it is inferrable from the output.
        assert idx >= 1
        # noinspection PyTypeChecker
        chi = np.abs((hm_out[idx][1] - hm_out[idx - 1][1]) / max(hm_out[idx][0] - hm_out[idx - 1][0], 1e-9))
        if H * direction >= H_stop_range[0] and chi < saturation_susceptibility:
            return ""
        if H * direction > H_stop_range[1]:
            return ""

    return "Maximum number of iterations reached"


@njit(nogil=True)
def _dM_dH(c_r: float, M_s: float, a: float, k_p: float, alpha: float, H: float, M: float, direction: int) -> float:
    # noinspection PyTypeChecker,PyShadowingNames
    """
    Evaluates the magnetic susceptibility derivative at the given point of the M(H) curve.
    The result is sensitive to the sign of the H change; the direction is defined as sign(dH).
    This implements the model described in "Jiles–Atherton Magnetic Hysteresis Parameters Identification", Pop et al.

    >>> fun = lambda H, M, d: _dM_dH(**dataclasses.asdict(COEF_COMSOL_JA_MATERIAL), H=H, M=M, direction=d)
    >>> assert np.isclose(fun(0, 0, +1), fun(0, 0, -1))
    >>> assert np.isclose(fun(+1, 0, +1), fun(-1, 0, -1))
    >>> assert np.isclose(fun(-1, 0, +1), fun(+1, 0, -1))
    >>> assert np.isclose(fun(-1, 0.8e6, +1), fun(+1, -0.8e6, -1))
    >>> assert np.isclose(fun(+1, 0.8e6, +1), fun(-1, -0.8e6, -1))
    """
    assert direction in (-1, +1)
    H_e = H + alpha * M
    M_an = M_s * _langevin(H_e / a)
    dM_an_dH_e = M_s / a * _dL_dx(H_e / a)
    dM_irr_dH = (M_an - M) / (k_p * direction * (1 - c_r) - alpha * (M_an - M))
    return (c_r * dM_an_dH_e + (1 - c_r) * dM_irr_dH) / (1 - alpha * c_r)  # type: ignore


@njit(nogil=True)
def _langevin(x: float) -> float:
    """
    L(x) = coth(x) - 1/x
    For tensors, the function is applied element-wise.
    """
    if np.abs(x) < 1e-10:  # For very small |x|, use the series expansion ~ x/3
        return x / 3.0
    return 1.0 / np.tanh(x) - 1.0 / x  # type: ignore


@njit(nogil=True)
def _dL_dx(x: float) -> float:
    """
    Derivative of Langevin L(x) = coth(x) - 1/x.
    d/dx [coth(x) - 1/x] = -csch^2(x) + 1/x^2.
    """
    if np.abs(x) < 1e-10:  # series expansion of L(x) ~ x/3 -> derivative ~ 1/3 near zero
        return 1.0 / 3.0
    # exact expression: -csch^2(x) + 1/x^2
    # csch^2(x) = 1 / sinh^2(x)
    return -1.0 / (np.sinh(x) ** 2) + 1.0 / (x**2)  # type: ignore


_logger = getLogger(__name__)
