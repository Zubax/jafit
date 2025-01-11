# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

"""
Implementation of the Jiles-Atherton model from COMSOL Multiphysics.
For the definition of the model, see the enclosed PDF with the relevant excerpt from the COMSOL user reference.
"""

from __future__ import annotations
import itertools
from logging import getLogger
import dataclasses
import numpy as np
import numpy.typing as npt
from .util import njit


mu_0 = 1.2566370614359173e-6  # Vacuum permeability [henry/meter]

_EPSILON = 1e-9  # Small number for numerical stability


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


COEF_COMSOL_JA_MATERIAL = Coef(c_r=0.1, M_s=1e6, a=560, k_p=1200, alpha=0.0007)
"""
Values are taken from the material properties named "Jiles-Atherton Hysteretic Material" in COMSOL Multiphysics.
Useful for testing and validation.
"""


@dataclasses.dataclass(frozen=True)
class Solution:
    """
    Each curve segment contains an nx3 matrix, where the columns are the applied field H, magnetization M,
    and flux density B, respectively; one row per sample point.
    """

    HMB_virgin: npt.NDArray[np.float64]
    HMB_major_descending: npt.NDArray[np.float64]
    HMB_major_ascending: npt.NDArray[np.float64] | None = None

    # TODO balancing: mirror the ascending curve around the origin and merge it with the descending curve
    # Requires interpolation.


def solve(
    coef: Coef,
    *,
    tolerance: float = 1e-3,
    saturation_susceptibility: float = 0.1,
    H_stop_range: tuple[float, float] = (100e3, 3e6),
    max_iter: int = 10**6,
    no_ascent: bool = False,
) -> Solution:
    """
    Solves the JA model for the given coefficients and initial conditions by sweeping the magnetization in the
    specified direction using the specified step size until saturation, and then retraces the magnetization back
    until saturation in the opposite direction.

    At the moment, only scalar definition is supported, but this may be extended if necessary.

    The sweeps will stop when either the magnetic susceptibility drops below the specified threshold (which indicates
    that the material is saturated) or when the applied field magnitude exceeds ```H_stop_range[1]```.
    The saturation will not be detected unless the H-field magnitude is at least ```H_stop_range[0]```;
    this is done to handle certain materials that exhibit very low susceptibility at weak fields.

    The solver will stop the sweep early if a floating point error is raised.
    This can be paired with ``np.seterr(over="raise")`` to simply utilize the maximum range of the floating point
    representation without any special handling.

    Returns sample points for the H, M, and B fields in the order of their appearance.
    """
    c_r, M_s, a, k_p, alpha = coef.c_r, coef.M_s, coef.a, coef.k_p, coef.alpha

    def sweep_hmb(H: float, M: float, sign: int) -> npt.NDArray[np.float64]:
        assert sign in (-1, +1)
        hmb = np.empty((max_iter, 3), dtype=np.float64)
        hmb[0] = H, M, np.nan  # The initial point shall be set by the caller.
        idx = np.array([0], dtype=np.uint32)
        try:
            status = _solve(
                hm_out=hmb[:, :2],
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
            H, M, _ = hmb[idx[0]]
            _logger.warning("Sweep stopped at #%s, H=%+.3f, M=%+.3f due to numerical error: %s", idx[0], H, M, ex)

        hmb = hmb[: idx[0] + 1]
        hmb[:, 2] = mu_0 * (hmb[:, 0] + hmb[:, 1])
        hmb.setflags(write=False)
        H, M, _ = hmb[-1]
        if status:
            raise ConvergenceError(f"Convergence failure at #{idx[0]}, H={H:+.3f}, M={M:+.3f}: {status}")
        return hmb

    hmb_virgin = sweep_hmb(0, 0, +1)

    H_last, M_last, _ = hmb_virgin[-1]
    hmb_maj_dsc = sweep_hmb(H_last, M_last, -1)

    hmb_maj_asc: npt.NDArray[np.float64] | None = None
    if not no_ascent:
        H_last, M_last, _ = hmb_maj_dsc[-1]
        hmb_maj_asc = sweep_hmb(H_last, M_last, +1)

    return Solution(HMB_virgin=hmb_virgin, HMB_major_descending=hmb_maj_dsc, HMB_major_ascending=hmb_maj_asc)


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

    dH_abs = 1e-3  # This is just a guess that will be dynamically refined.
    max_iter = hm_out.shape[0]
    assert np.isfinite(hm_out[0]).all()  # We require that the first point is already filled in.
    while idx_out[0] < max_iter:
        dH = direction * dH_abs
        H, M = hm_out[idx_out[0]]

        # Integrate using Heun's method (instead of Euler's) for better stability.
        # _dM_dH() may throw FloatingPointError or ZeroDivisionError; we can't handle them in the compiled code.
        # Since we always update the current state in the caller's context, we can still return partial solution
        # even if an exception is raised.
        dM_dH_1 = _dM_dH(c_r=c_r, M_s=M_s, a=a, k_p=k_p, alpha=alpha, H=H, M=M, direction=direction)
        M_1 = M + dM_dH_1 * dH
        dM_dH_2 = _dM_dH(c_r=c_r, M_s=M_s, a=a, k_p=k_p, alpha=alpha, H=H + dH, M=M_1, direction=direction)
        M_2 = M + 0.5 * (dM_dH_1 + dM_dH_2) * dH

        # Check if the tolerance is acceptable, only accept the data point if it is; otherwise refine and retry.
        if np.abs(M_1 - M_2) > tolerance:
            dH_abs *= 0.5
            if dH_abs < 1e-8:
                return "Convergence failure: error still large at the smallest step size"
            continue

        # Save the next point. This ensures that the caller will get partial results even if an exception is raised.
        idx = idx_out[0] + 1
        idx_out[0] = idx
        dH_abs = min(dH_abs * 1.1, 1e3)
        hm_out[idx] = H + dH, M_2

        # Termination check and logging. It is guaranteed that we have at least two points now.
        # No need to log the termination reason here because it is easily inferrable from the output.
        chi = np.abs((hm_out[idx][1] - hm_out[idx - 1][1]) / (hm_out[idx][0] - hm_out[idx - 1][0]))
        if H * direction >= H_stop_range[0] and chi < saturation_susceptibility:
            return ""
        if H * direction > H_stop_range[1]:
            return ""

    return "Maximum number of iterations reached"


@njit(nogil=True)
def _dM_dH(c_r: float, M_s: float, a: float, k_p: float, alpha: float, H: float, M: float, direction: int) -> float:
    # noinspection PyTypeChecker
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
    if np.abs(x) < _EPSILON:  # For very small |x|, use the series expansion ~ x/3
        return x / 3.0
    return 1.0 / np.tanh(x) - 1.0 / x  # type: ignore


@njit(nogil=True)
def _dL_dx(x: float) -> float:
    """
    Derivative of Langevin L(x) = coth(x) - 1/x.
    d/dx [coth(x) - 1/x] = -csch^2(x) + 1/x^2.
    """
    if np.abs(x) < _EPSILON:  # series expansion of L(x) ~ x/3 -> derivative ~ 1/3 near zero
        return 1.0 / 3.0
    # exact expression: -csch^2(x) + 1/x^2
    # csch^2(x) = 1 / sinh^2(x)
    return -1.0 / (np.sinh(x) ** 2) + 1.0 / (x**2)  # type: ignore


_logger = getLogger(__name__)
