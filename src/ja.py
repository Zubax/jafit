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


mu_0 = 1.2566370614359173e-6  # Vacuum permeability [henry/meter]
EPSILON = 1e-9  # Small number for numerical stability


@dataclasses.dataclass
class Coeffs:
    """
    Per the original model definition, all of them can be scalars or 3x3 matrices:

        Symbol  Description                                                         Range           Unit
        c_r     magnetization reversibility (1 for purely anhysteretic material)    (0, 1]          dimensionless
        M_s     saturation magnetization                                            positive real   ampere/meter
        a       domain wall density                                                 positive real   ampere/meter
        k_p     pinning loss                                                        positive real   ampere/meter
        alpha   interdomain coupling                                                non-negative    dimensionless
    """

    c_r: float
    M_s: float
    a: float
    k_p: float
    alpha: float

    def __post_init__(self) -> None:
        def is_positive_real(x: float) -> bool:
            return x > 0 and np.isfinite(x)

        if not 0 < self.c_r <= 1:
            raise ValueError(f"c_r invalid: {self.c_r}")
        if not is_positive_real(self.M_s):
            raise ValueError(f"M_s invalid: {self.M_s}")
        if not is_positive_real(self.a):
            raise ValueError(f"a invalid: {self.a}")
        if not is_positive_real(self.k_p):
            raise ValueError(f"k_p invalid: {self.k_p}")
        if self.alpha < 0:
            raise ValueError(f"alpha invalid: {self.alpha}")


COEF_INITIAL = Coeffs(
    c_r=0.1,
    M_s=1e6,
    a=560,
    k_p=1200,
    alpha=0.0007,
)
"""
These starting values are taken from the material properties named "Jiles-Atherton Hysteretic Material"
in COMSOL Multiphysics. They can (and should) be refined further to speedup convergence.
"""


@dataclasses.dataclass(frozen=True)
class Solution:
    H_M_B_segments: list[npt.NDArray[np.float64]]
    """
    A list of hysteresis loop fragments in the order of appearance.
    The first fragment is from zero to forward saturation, then to reverse saturation, then back forward.
    This results in the construction of the full hysteresis loop starting from the virgin curve.

    Each fragment contains an nx3 matrix, where the columns are the applied field H, magnetization M,
    and flux density B, respectively; one row per sample point.
    """


# noinspection PyPep8Naming
def solve(
    coef: Coeffs,
    *,
    H_step: float,
    dM_dH_saturation_threshold: float,
    H_magnitude_limit: float,
) -> Solution:
    """
    Solves the JA model for the given coefficients and initial conditions by sweeping the magnetization in the
    specified direction using the specified step size until saturation, and then retraces the magnetization back
    until saturation in the opposite direction.

    At the moment, only scalar definition is supported, but this may be extended if necessary.

    Returns sample points for the H, M, and B fields in the order of their appearance.
    """
    H_step = float(H_step)
    if not 0 < H_step or not np.isfinite(H_step):
        raise ValueError(f"H_step invalid: {H_step}")

    hmb_fragments: list[list[tuple[float, float, float]]] = []

    # noinspection PyPep8Naming
    def sweep_to_saturation(H: float, M: float, sign: int) -> list[tuple[float, float, float]]:
        assert sign in (-1, +1)
        out: list[tuple[float, float, float]] = []
        for idx in itertools.count():
            H_old, M_old = H, M

            # Integrate this step. We should be using a proper ODE solver here.
            dM = _dM_dH(coef, H=H, M=M, dH_diff_step=max(H_step * 1e-3, 1e-6))
            H += sign * H_step
            M += dM * (sign * H_step)
            B = mu_0 * (H + M)

            # Post-process the new data point.
            out.append((H, M, B))
            dM_dH_numeric = (M - M_old) / (H - H_old)
            if (sign > 0) == (H > 0) and dM_dH_numeric < dM_dH_saturation_threshold:
                _logger.info(f"Sweep stopped at H={H:+.3f}, M={M:+.3f} due to saturation")
                break
            if np.abs(H) > H_magnitude_limit:
                _logger.info(f"Sweep stopped at H={H:+.3f}, M={M:+.3f} due to H magnitude limit")
                break
            if idx % 10000 == 0:
                _logger.info(f"{('','↑', '↓')[sign]}#{idx}: H={H:+.3f}, M={M:+.3f}, dM/dH={dM_dH_numeric:.6f}")
        return out

    # Virgin curve to positive saturation
    hmb_fragments.append(sweep_to_saturation(0, 0, +1))

    # Reverse sweep to negative saturation
    H_last, M_last = hmb_fragments[-1][-1][0], hmb_fragments[-1][-1][1]
    hmb_fragments.append(sweep_to_saturation(H_last, M_last, -1))

    # Forward sweep to positive saturation to complete the major loop
    H_last, M_last = hmb_fragments[-1][-1][0], hmb_fragments[-1][-1][1]
    hmb_fragments.append(sweep_to_saturation(H_last, M_last, +1))

    return Solution(
        H_M_B_segments=[np.array(x, dtype=np.float64) for x in hmb_fragments],
    )


# noinspection PyPep8Naming
def _dM_dH(coef: Coeffs, *, H: float, M: float, dH_diff_step: float) -> float:
    """
    Returns dM/dH for the 1D Jiles–Atherton model, given the current applied field and magnetization.

    >>> fun = lambda H, M: _dM_dH(COEF_INITIAL, H=H, M=M, dH_diff_step=1e-3)
    >>> assert np.isclose(fun(-1, 0), fun(+1, 0))
    >>> assert np.isclose(fun(-1, 0.8e6), fun(+1, -0.8e6))
    >>> assert np.isclose(fun(+1, 0.8e6), fun(-1, -0.8e6))
    """
    H_e = H + coef.alpha * M
    M_an = coef.M_s * _langevin(np.abs(H_e) / coef.a) * np.sign(H_e)
    # Derivative of anhysteretic magnetization M_an wrt effective H-field H_e.
    # Derived from the above with simplification: langevin(abs(H_e)/a)*sign(H_e) => langevin(H_e/a).
    dM_an_dH_e = coef.M_s / coef.a * _dL_dx(H_e / coef.a)

    # The key equation of the J-A model is:
    #   dM = max(x*dH_e,0)*sign(x) + c_r*dM_an
    # where:
    #   x = k_p**-1 * (M_an-M)
    # This form is highly unalgebraic due to the discontinuous max() and sign(),
    # so we approximate the differential with finite differences.
    # noinspection PyPep8Naming
    def dM(dH: float) -> float:
        x = (M_an - M) / coef.k_p
        dM_an = dM_an_dH_e * dH
        return float(np.max(x * dH, 0) * np.sign(x) + coef.c_r * dM_an)

    dM_dH_e = (dM(+dH_diff_step) - dM(-dH_diff_step)) / (2 * dH_diff_step)

    # To find dM/dH, recall that H_e = H + alpha*M, so:
    #   dM/dH = (dM/dH_e) * (1 - alpha*dM/dH_e)
    # Singularities afoot when alpha*dM/dH_e = 1, which requires special handling.
    if np.abs(1 - coef.alpha * dM_dH_e) < EPSILON:
        _logger.warning(f"Singularity in dM/dH at H={H}, M={M}, H_e={H_e}, dM/dH_e={dM_dH_e}, coef={coef}")
        return 0.0  # Avoid division by zero in pathological cases
    return float(dM_dH_e / (1 - coef.alpha * dM_dH_e))


def _langevin(x: float) -> float:
    """
    L(x) = coth(x) - 1/x
    For tensors, the function is applied element-wise.
    """
    if np.abs(x) < EPSILON:  # For very small |x|, use the series expansion ~ x/3
        return x / 3.0
    return float(1.0 / np.tanh(x) - 1.0 / x)


# noinspection PyPep8Naming
def _dL_dx(x: float) -> float:
    """
    Derivative of Langevin L(x) = coth(x) - 1/x.
    d/dx [coth(x) - 1/x] = -csch^2(x) + 1/x^2.
    """
    if np.abs(x) < EPSILON:  # series expansion of L(x) ~ x/3 -> derivative ~ 1/3 near zero
        return 1.0 / 3.0
    # exact expression: -csch^2(x) + 1/x^2
    # csch^2(x) = 1 / sinh^2(x)
    return float(-1.0 / (np.sinh(x) ** 2) + 1.0 / (x**2))


_logger = getLogger(__name__)
