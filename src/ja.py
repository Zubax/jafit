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

_EPSILON = 1e-9  # Small number for numerical stability


@dataclasses.dataclass(frozen=True)
class Coef:
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

        if not 0 <= self.c_r <= 1:
            raise ValueError(f"c_r invalid: {self.c_r}")
        if not is_positive_real(self.M_s):
            raise ValueError(f"M_s invalid: {self.M_s}")
        if not is_positive_real(self.a):
            raise ValueError(f"a invalid: {self.a}")
        if not is_positive_real(self.k_p):
            raise ValueError(f"k_p invalid: {self.k_p}")
        if self.alpha < 0:
            raise ValueError(f"alpha invalid: {self.alpha}")


COEF_COMSOL_JA_MATERIAL = Coef(
    c_r=0.1,
    M_s=1e6,
    a=560,
    k_p=1200,
    alpha=0.0007,
)
"""
Values are taken from the material properties named "Jiles-Atherton Hysteretic Material" in COMSOL Multiphysics.
Usefuil for testing and validation.
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
    coef: Coef,
    *,
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
    hmb_fragments: list[list[tuple[float, float, float]]] = []

    # noinspection PyPep8Naming
    def sweep(H: float, M: float, sign: int) -> list[tuple[float, float, float]]:
        assert sign in (-1, +1)
        hm: list[tuple[float, float]] = []
        dH_abs = 10  # TODO FIXME
        for idx in itertools.count():
            # Save the state first because we want to keep the initial point.
            hm.append((H, M))

            # Integrate using Heun's method (instead of Euler's) for better stability.
            dH = sign * dH_abs
            dM_dH_1 = _dM_dH(coef, H=H, M=M, direction=sign)
            dM_dH_2 = _dM_dH(coef, H=H + dH, M=M + dM_dH_1 * dH, direction=sign)
            M = M + 0.5 * (dM_dH_1 + dM_dH_2) * dH
            # M = M + dM_1 * dH  # Euler's forward method
            H = H + dH

            # Termination check and logging.
            dM_dH_numeric = ((hm[-1][1] - hm[-2][1]) / (hm[-1][0] - hm[-2][0])) if len(hm) > 1 else np.inf
            if (sign > 0) == (H > 0) and dM_dH_numeric < dM_dH_saturation_threshold:
                _logger.info(f"Sweep stopped at H={H:+.3f}, M={M:+.3f} due to saturation")
                break
            if np.abs(H) > H_magnitude_limit:
                _logger.info(f"Sweep stopped at H={H:+.3f}, M={M:+.3f} due to H magnitude limit")
                break
            if idx % 10000 == 0:
                _logger.info(f"{('','↑', '↓')[sign]}#{idx}: H={H:+.3f}, M={M:+.3f}, dM/dH={dM_dH_numeric:.6f}")
        return [(h, m, mu_0 * (h + m)) for h, m in hm]

    # Virgin curve to positive saturation
    hmb_fragments.append(sweep(0, 0, +1))

    # Reverse sweep to negative saturation
    H_last, M_last = hmb_fragments[-1][-1][0], hmb_fragments[-1][-1][1]
    hmb_fragments.append(sweep(H_last, M_last, -1))

    # Forward sweep to positive saturation to complete the major loop
    H_last, M_last = hmb_fragments[-1][-1][0], hmb_fragments[-1][-1][1]
    hmb_fragments.append(sweep(H_last, M_last, +1))

    return Solution(
        H_M_B_segments=[np.array(x, dtype=np.float64) for x in hmb_fragments],
    )


# noinspection PyPep8Naming
def _dM_dH(coef: Coef, *, H: float, M: float, direction: int) -> float:
    """
    Evaluates the magnetic susceptibility derivative at the given point of the M(H) curve.
    The result is sensitive to the sign of the H change; the direction is defined as sign(dH).
    This implements the model described in "Jiles–Atherton Magnetic Hysteresis Parameters Identification", Pop et al.

    >>> fun = lambda H, M, d: _dM_dH(COEF_COMSOL_JA_MATERIAL, H=H, M=M, direction=d)
    >>> assert np.isclose(fun(-1, 0, +1), fun(+1, 0, +1))
    >>> assert np.isclose(fun(-1, 0.8e6, +1), fun(+1, -0.8e6, +1))
    >>> assert np.isclose(fun(+1, 0.8e6, +1), fun(-1, -0.8e6, +1))
    """
    if direction not in (-1, +1):
        raise ValueError(f"Invalid direction: {direction}")

    H_e = H + coef.alpha * M
    M_an = coef.M_s * _langevin(H_e / coef.a)
    dM_an_dH_e = coef.M_s / coef.a * _dL_dx(H_e / coef.a)
    dM_irr_dH = (M_an - M) / (coef.k_p * direction * (1 - coef.c_r) - coef.alpha * (M_an - M))
    return (coef.c_r * dM_an_dH_e + (1 - coef.c_r) * dM_irr_dH) / (1 - coef.alpha * coef.c_r)


def _langevin(x: float) -> float:
    """
    L(x) = coth(x) - 1/x
    For tensors, the function is applied element-wise.
    """
    if np.abs(x) < _EPSILON:  # For very small |x|, use the series expansion ~ x/3
        return x / 3.0
    return float(1.0 / np.tanh(x) - 1.0 / x)


# noinspection PyPep8Naming
def _dL_dx(x: float) -> float:
    """
    Derivative of Langevin L(x) = coth(x) - 1/x.
    d/dx [coth(x) - 1/x] = -csch^2(x) + 1/x^2.
    """
    if np.abs(x) < _EPSILON:  # series expansion of L(x) ~ x/3 -> derivative ~ 1/3 near zero
        return 1.0 / 3.0
    # exact expression: -csch^2(x) + 1/x^2
    # csch^2(x) = 1 / sinh^2(x)
    return float(-1.0 / (np.sinh(x) ** 2) + 1.0 / (x**2))


_logger = getLogger(__name__)
