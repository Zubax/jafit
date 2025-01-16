# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

"""
Jiles-Atherton solver.
"""

from __future__ import annotations
from logging import getLogger
import dataclasses
import numpy as np
import numpy.typing as npt
import scipy.integrate
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
        c_r     magnetization reversibility (1 for purely anhysteretic material)    (0, 1]          dimensionless
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

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"c_r={self.c_r:.18f}, "
            f"M_s={self.M_s:017.9f}, "
            f"a={self.a:016.9f}, "
            f"k_p={self.k_p:016.9f}, "
            f"alpha={self.alpha:.18f})"
        )


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


MAX_COARSENESS = 2


def solve(
    coef: Coef,
    *,
    saturation_susceptibility: float = 0.05,
    H_stop_range: tuple[float, float] = (100e3, 3e6),
    no_balancing: bool = False,
    coarseness: int = 0,
) -> Solution:
    """
    Solves the JA model for the given coefficients by sweeping the magnetization in the specified direction
    until saturation, then retraces the magnetization back until saturation in the opposite direction,
    and then forward again to reconstruct the full major loop.

    At the moment, only scalar definition is supported, but this may be extended if necessary.

    The sweeps will stop when either the magnetic susceptibility drops below the specified threshold (which indicates
    that the material is saturated) or when the applied field magnitude exceeds ```H_stop_range[1]```.
    The saturation will not be detected unless the H-field magnitude is at least ```H_stop_range[0]```;
    this is done to handle materials that exhibit very low susceptibility at weak fields.
    """

    def do_sweep(H0: float, M0: float, sign: int) -> npt.NDArray[np.float64]:
        def once() -> npt.NDArray[np.float64]:
            return _sweep(
                coef,
                H0=H0,
                M0=M0,
                sign=sign,
                saturation_susceptibility=saturation_susceptibility,
                H_stop_range=H_stop_range,
                coarseness=crs,
            )

        crs = coarseness
        while True:
            try:
                return once()
            except ConvergenceError as ex:
                if crs >= MAX_COARSENESS:
                    raise
                crs += 1
                _logger.debug("Retrying the sweep with coarseness %d due to %s: %s", crs, type(ex).__name__, ex)

    hm_virgin = do_sweep(0, 0, +1)

    H_last, M_last = hm_virgin[-1]
    hm_maj_dsc = do_sweep(H_last, M_last, -1)

    H_last, M_last = hm_maj_dsc[-1]
    hm_maj_asc = do_sweep(H_last, M_last, +1)

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


def _sweep(
    coef: Coef,
    *,
    H0: float,
    M0: float,
    sign: int,
    saturation_susceptibility: float,
    H_stop_range: tuple[float, float],
    coarseness: int,
) -> npt.NDArray[np.float64]:
    assert sign in (-1, +1)
    if not (0 <= coarseness <= MAX_COARSENESS):
        raise ValueError(f"Invalid coarseness: {coarseness}")
    c_r, M_s, a, k_p, alpha = coef.c_r, coef.M_s, coef.a, coef.k_p, coef.alpha

    def rhs(x: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        z = _dM_dH(c_r=c_r, M_s=M_s, a=a, k_p=k_p, alpha=alpha, H=x, M=float(y[0]), direction=sign)
        return np.array([z], dtype=np.float64)

    # Initially, I tried solving this equation using explicit methods, specifically RK4 and Dormand-Prince (RK45)
    # (see the git history for details).
    # However, the equation can be quite stiff at certain coefficients that do occur in practice with some softer
    # materials, like certain AlNiCo grades, which causes explicit solvers to diverge even at very fine steps.
    # Hence, it appears to be necessary to employ implicit methods.
    # noinspection PyUnresolvedReferences
    solver = scipy.integrate.BDF(
        rhs,
        H0,
        np.array([M0], dtype=np.float64),
        t_bound=H_stop_range[1] * sign,
        max_step=(1e2, 1e3, 1e4)[coarseness],
        rtol=(0.001, 0.01, 1.0)[coarseness],  # Takes precedence at strong magnetization
        atol=(0.1, 1.0, 1000)[coarseness],  # Takes precedence at weak magnetization
    )
    hm = np.empty((10**8, 2), dtype=np.float64)
    hm[0] = H0, M0
    idx = 1
    while solver.status == "running":
        H_old, M_old = solver.t, solver.y[0]
        try:
            msg = solver.step()
        except (ZeroDivisionError, FloatingPointError) as ex:
            msg = f"#{idx*sign:+d} {H0=} {M0=} H={H_old} M={M_old}: {type(ex).__name__}: {ex}"
            if idx < 10:
                raise NumericalError(msg) from ex
            _logger.warning("Stopping the sweep early: %s", msg)
            _logger.debug("Stack trace for the above error", exc_info=True)
            break
        if solver.status not in ("running", "finished"):
            raise ConvergenceError(
                f"#{idx*sign:+06d} {H0=:+012.3f} {M0=:+012.3f} H={H_old:+012.3f} M={M_old:+012.3f}: {msg}"
            )
        H_new, M_new = solver.t, solver.y[0]
        hm[idx] = H_new, M_new
        idx += 1
        # Terminate sweep early if the material is already saturated.
        chi = abs((M_new - M_old) / (H_new - H_old)) if abs(H_new - H_old) > 1e-10 else np.inf
        if H_new * sign >= H_stop_range[0] and chi < saturation_susceptibility:
            break

    return hm[:idx]


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
