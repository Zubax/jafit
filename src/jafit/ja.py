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
from .util import njit, relative_distance


class SolverError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        partial_curves: list[npt.NDArray[np.float64]] | None = None,
    ) -> None:
        super().__init__(message)
        self.partial_curves = partial_curves or []


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

    For soft materials, k_p is approximately equal to the intrinsic coercivity H_ci.
    During fitting, it is a good idea to start with k_p≈H_c.

    Large values of alpha may undermine the numerical stability of the solver because it may cause small denominators
    to occur during the dM/dH computation.
    """

    c_r: float
    M_s: float
    a: float
    k_p: float
    alpha: float

    def __post_init__(self) -> None:
        def non_negative(x: float) -> bool:
            return x >= 0 and np.isfinite(x)

        if not 0 < self.c_r <= 1:
            raise ValueError(f"c_r invalid: {self.c_r}")
        if not non_negative(self.M_s):
            raise ValueError(f"M_s invalid: {self.M_s}")
        if not non_negative(self.a):
            raise ValueError(f"a invalid: {self.a}")
        if not non_negative(self.k_p):
            raise ValueError(f"k_p invalid: {self.k_p}")
        if not non_negative(self.alpha):
            raise ValueError(f"alpha invalid: {self.alpha}")
        if np.isclose(self.c_r * self.alpha, 1):
            _logger.warning(
                "c_r*alpha≈1; no solutions exist for this combination of coefficients: c_r=%f alpha=%f",
                self.c_r,
                self.alpha,
            )

    def __str__(self) -> str:
        return (
            f"{type(self).__name__}("
            f"c_r={self.c_r:.17f}, "
            f"M_s={self.M_s:017.9f}, "
            f"a={self.a:016.9f}, "
            f"k_p={self.k_p:016.9f}, "
            f"alpha={self.alpha:.17f})"
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
    All curve segments are ordered in the ascending order of the H-field.
    """

    virgin: npt.NDArray[np.float64]
    descending: npt.NDArray[np.float64]
    ascending: npt.NDArray[np.float64]


def solve(
    coef: Coef,
    H_stop: tuple[float, float] | float,
    *,
    saturation_susceptibility: float = 0.05,
    fast: bool = False,
) -> Solution:
    """
    Solves the JA model for the given coefficients by sweeping the magnetization in the specified direction
    until saturation, then retraces the magnetization back until saturation in the opposite direction,
    and then forward again to reconstruct the full loop.

    At the moment, only scalar definition is supported, but this may be extended if necessary.

    The sweeps will stop when either the magnetic susceptibility drops below the specified threshold (which indicates
    that the material is saturated) or when the applied field magnitude exceeds ``H_stop[1]``.
    The saturation will not be detected unless the H-field magnitude is at least ``H_stop[0]``;
    this is done to handle materials that exhibit very low susceptibility at weak fields.
    If ``H_stop`` is a scalar, the sweep will have a fixed H amplitude of that value.
    If the ``H_stop`` amplitude is not enough to saturate the material, the resulting loop will be a minor one.

    Some of the coefficients sets result in great stiffness;
    they are provided here for testing and illustration purposes.
    The solver is expected to manage that by dynamically adjusting the tolerance bounds.

    >>> stiff = Coef(c_r=0.956886485630692230, M_s=2956870.912007416, a=025069.875361107, k_p=019498.206218542, alpha=0.181220196232252662)
    >>> solve(stiff, (100e3, 3e6))  # doctest: +ELLIPSIS
    Solution(...)

    >>> stiff = Coef(c_r=0.1, M_s=1191941.07155, a=65253, k_p=85677, alpha=0.19)
    >>> solve(stiff, (100e3, 3e6))  # doctest: +ELLIPSIS
    Solution(...)
    """
    if fast:
        _logger.debug("Fast mode is enabled; the solver may produce inaccurate results or fail to converge.")

    if isinstance(H_stop, float):
        H_stop_range = H_stop, H_stop
    else:
        H_stop_range = float(H_stop[0]), float(H_stop[1])
    if not len(H_stop_range) == 2 or not np.isfinite(H_stop_range).all() or not np.all(np.array(H_stop_range) > 0):
        raise ValueError(f"Invalid H stop range: {H_stop_range}")

    def do_sweep(H0: float, M0: float, sign: int) -> npt.NDArray[np.float64]:
        try:
            return _sweep(
                coef,
                H0=H0,
                M0=M0,
                sign=sign,
                saturation_susceptibility=saturation_susceptibility,
                H_stop_range=H_stop_range,
                fast=fast,
            )
        except Exception:
            _logger.debug("Terminating sweep: %s sign=%d H0=%+f M0=%+f", coef, sign, H0, M0)
            raise

    hm_virgin = do_sweep(0, 0, +1)

    H_last, M_last = hm_virgin[-1]
    try:
        hm_dsc = do_sweep(H_last, M_last, -1)
    except SolverError as ex:
        ex.partial_curves = [hm_virgin] + ex.partial_curves
        raise ex

    H_last, M_last = hm_dsc[-1]
    try:
        hm_asc = do_sweep(H_last, M_last, +1)
    except SolverError as ex:
        ex.partial_curves = [hm_virgin, hm_dsc] + ex.partial_curves
        raise ex

    return Solution(virgin=hm_virgin, descending=hm_dsc[::-1], ascending=hm_asc)


def _sweep(
    coef: Coef,
    *,
    H0: float,
    M0: float,
    sign: int,
    saturation_susceptibility: float,
    H_stop_range: tuple[float, float],
    fast: bool,
    save_delta: npt.NDArray[np.float64] = np.array([3.0, 1.0], dtype=np.float64),
    tolerance_loosening_factor: float = 10.0,
    worst_relative_tolerance: float = 0.01,
) -> npt.NDArray[np.float64]:
    assert sign in (-1, +1)
    assert save_delta.shape == (2,)
    assert 0 < worst_relative_tolerance < 1
    H_bound = H_stop_range[1] * sign

    # Initially, I tried solving this equation using explicit methods, specifically RK4 and Dormand-Prince (RK45)
    # (see the git history for details).
    # However, the equation can be quite stiff at certain coefficients that do occur in practice with some softer
    # materials, like certain AlNiCo grades, which causes explicit solvers to diverge even at very fine steps.
    # Hence, it appears to be necessary to employ implicit methods. Note though that we sweep H back and forth
    # manually, so our overall approach perhaps should be called semi-implicit.
    #
    # Here we implement an adaptive tolerance method which I am very pleased with: we start out with a good tolerance
    # setting and try solving the equation. If the solver fails to converge due to stiffness at some point,
    # we loosen the tolerance a little and try to continue. If it fails again, we worsen the tolerance further and
    # rollback to the last known good point achieved with the original best tolerance.
    # The reason we need the rollback is that every time we switch the settings we discard the solver state,
    # which may result in discontinuities in the solution, so it is desirable to keep the switching points to
    # the minimum, which is one.
    #
    # Ideally, we should be able to adjust the tolerance on the go without discarding the solver, but the SciPy API
    # does not seem to support this, and I don't have the time to roll out my own implicit solver.
    rtol, atol = 1e-6, 1e-4
    max_step = min(200.0, H_stop_range[0] / 200, H_stop_range[1] / 1000)
    if fast:
        rtol, atol, max_step = [x * 10 for x in (rtol, atol, max_step)]
    _logger.debug(
        "Starting sweep: H=[%+.f→%+.f] M0=%+.f tol=(%.1e×M+%.1e) max_step=%.1f",
        H0,
        H_bound,
        M0,
        rtol,
        atol,
        max_step,
    )
    solver = _make_solver(coef, H0, M0, H_bound, rtol=rtol, atol=atol, max_step=max_step)
    hm = np.empty((10**7, 2), dtype=np.float64)
    hm[0] = H0, M0
    idx = 1
    checkpoint: tuple[int, float, float] | None = None
    while solver.status == "running":
        H_old, M_old = solver.t, solver.y[0]
        try:
            # Radau has a bug in predict_factor() that causes a division by zero; it is benign though and can be ignored
            with np.errstate(divide="ignore"):
                msg = solver.step()
        except (ZeroDivisionError, FloatingPointError) as ex:
            _logger.debug("Stack trace for %s", type(ex).__name__, exc_info=True)
            raise NumericalError(f"#{idx*sign:+d} {H0=} {M0=} H={H_old} M={M_old}: {type(ex).__name__}: {ex}") from ex

        if solver.status not in ("running", "finished"):
            _logger.debug(
                "Solver failure at idx=%+07d H=%+018.9f M=%+018.9f with rtol=%.1e atol=%.1e."
                " Rollback checkpoint is %s."
                " Error message was: %s",
                idx * sign,
                H_old,
                M_old,
                rtol,
                atol,
                checkpoint,
                msg,
            )
            rtol *= tolerance_loosening_factor
            atol *= tolerance_loosening_factor
            if rtol > worst_relative_tolerance:  # Give up, it's hopeless
                raise ConvergenceError(
                    f"#{idx * sign:+06d} {H0=:+012.3f} {M0=:+012.3f} H={H_old:+012.3f} M={M_old:+012.3f}: {msg}",
                    partial_curves=[hm[:idx]],
                )
            if checkpoint is None:
                checkpoint = idx, float(H_old), float(M_old)  # If it fails again, rollback to this point
            else:
                idx, H_old, M_old = checkpoint
            solver = _make_solver(coef, H_old, M_old, H_bound, rtol=rtol, atol=atol, max_step=max_step)
            continue

        mew = solver.t, solver.y[0]
        if np.any(np.abs(hm[idx - 1] - mew) > save_delta):
            hm[idx] = mew
            idx += 1
        H_new, M_new = mew

        # Terminate sweep early if the material is already saturated.
        chi = abs((M_new - M_old) / (H_new - H_old)) if abs(H_new - H_old) > 1e-10 else np.inf
        if H_new * sign >= H_stop_range[0] and chi < saturation_susceptibility:
            break

    # Ensure the last point is saved.
    if np.abs(hm[idx - 1] - (mew := solver.t, solver.y[0])).max() > 1e-9:
        hm[idx] = mew
        idx += 1

    if checkpoint is not None:
        _logger.debug("Solved with degraded tolerance due to great stiffness: rtol=%.1e atol=%.1e", rtol, atol)

    return hm[:idx]


# noinspection PyUnresolvedReferences
def _make_solver(
    coef: Coef,
    H0: float,
    M0: float,
    H_bound: float,
    rtol: float,
    atol: float,
    max_step: float,
) -> scipy.integrate.OdeSolver:
    """
    tolerance=rtol*M+atol; rtol dominates at strong magnetization, atol dominates at weak magnetization.
    """
    c_r, M_s, a, k_p, alpha = coef.c_r, coef.M_s, coef.a, coef.k_p, coef.alpha
    sign = int(np.sign(H_bound))

    def rhs(x: float, y: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        z = _dM_dH(c_r=c_r, M_s=M_s, a=a, k_p=k_p, alpha=alpha, H=x, M=float(y[0]), sign=sign)
        return np.array([z], dtype=np.float64)

    first_step = np.finfo(np.float64).eps * 100 * np.abs((H0, M0, 1)).max()
    assert first_step > np.finfo(np.float64).eps
    assert np.nextafter(H0, H0 + first_step * sign) != H0

    return scipy.integrate.Radau(  # Radau seems to be marginally more stable than BDF?
        rhs,
        H0,
        np.array([M0], dtype=np.float64),
        first_step=first_step,
        t_bound=H_bound,
        max_step=max_step,  # Max step must be small always; larger steps are more likely to blow up
        rtol=rtol,  # Dominates at strong magnetization; must be <0.01 to avoid bad solutions
        atol=atol,  # Dominates at weak magnetization; must be <0.1 to avoid bad solutions
    )


@njit
def _dM_dH(c_r: float, M_s: float, a: float, k_p: float, alpha: float, H: float, M: float, sign: int) -> float:
    # noinspection PyTypeChecker,PyShadowingNames
    """
    Evaluates the magnetic susceptibility derivative at the given point of the M(H) curve.
    The result is sensitive to sign(dH).

    >>> fun = lambda H, M, d: _dM_dH(**dataclasses.asdict(COEF_COMSOL_JA_MATERIAL), H=H, M=M, direction=d)
    >>> assert np.isclose(fun(0, 0, +1), fun(0, 0, -1))
    >>> assert np.isclose(fun(+1, 0, +1), fun(-1, 0, -1))
    >>> assert np.isclose(fun(-1, 0, +1), fun(+1, 0, -1))
    >>> assert np.isclose(fun(-1, 0.8e6, +1), fun(+1, -0.8e6, -1))
    >>> assert np.isclose(fun(+1, 0.8e6, +1), fun(-1, -0.8e6, -1))
    """
    assert sign in (-1, +1)
    H_e = H + alpha * M
    M_an = M_s * _langevin(H_e / a)
    dM1 = M_an - M
    dM1 = max(dM1, 0.0) if sign > 0 else min(dM1, 0.0)
    dM2 = (1 + c_r) * (sign * k_p - alpha * (M_an - M))
    dM_an = M_s * _dL_dx(H_e / a) / a
    dM3 = c_r / (1 + c_r) * dM_an
    return dM1 / dM2 + dM3


@njit
def _langevin(x: float) -> float:
    """
    L(x) = coth(x) - 1/x
    For tensors, the function is applied element-wise.
    """
    if np.abs(x) < 1e-4:  # For very small |x|, use the series expansion ~ x/3
        return x / 3.0
    return 1.0 / np.tanh(x) - 1.0 / x  # type: ignore


@njit
def _dL_dx(x: float) -> float:
    """
    Derivative of Langevin L(x) = coth(x) - 1/x.
    d/dx [coth(x) - 1/x]  =  1 - coth(x)^2 + 1/x^2  =  -1/sinh(x)^2 + 1/x^2
    """
    # Danger! The small-value threshold has to be large here because the subsequent full form is very sensitive!
    if np.abs(x) < 1e-4:  # series expansion of L(x) ~ x/3 -> derivative ~ 1/3 near zero
        return 1.0 / 3.0
    if np.abs(x) > 355:
        # sinh(x) overflows float64 at x>710, sinh(x)**2 overflows at x>355;
        # in this case, apply approximation: the first term vanishes as -1/inf=0
        return 1.0 / (x**2)
    return -1.0 / (np.sinh(x) ** 2) + 1.0 / (x**2)  # type: ignore


_logger = getLogger(__name__)
