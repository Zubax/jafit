#!/usr/bin/env python3
# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

"""
This is a simple utility that fits the Jiles-Atherton (JA) model coefficients for a given BH curve.
The JA model used here follows that of COMSOL Multiphysics; see the enclosed PDF with the relevant excerpt
from the COMSOL user reference.

The following coefficients are defined; per the model definition, all of them can be scalars or 3x3 matrices:

Symbol  Description                                                         Range           Unit
c_r     magnetization reversibility (1 for purely anhysteretic material)    (0, 1]          dimensionless
M_s     saturation magnetization                                            positive real   ampere/meter
a       domain wall density                                                 positive real   ampere/meter
k_p     pinning loss                                                        positive real   ampere/meter
alpha   interdomain coupling                                                non-negative    dimensionless
"""

from __future__ import annotations
import copy
from logging import getLogger, basicConfig
import dataclasses
import numpy as np
import numpy.typing as npt


mu_0 = 1.2566370614359173e-6  # Vacuum permeability [henry/meter]
EPSILON = 1e-9  # Small number for numerical stability


@dataclasses.dataclass
class JAScalarCoeffs:
    c_r: float
    """Magnetization reversibility (1 for purely anhysteretic material) [dimensionless]"""
    M_s: float
    """Saturation magnetization [ampere/meter]"""
    a: float
    """Domain wall density [ampere/meter]"""
    k_p: float
    """Pinning loss [ampere/meter]"""
    alpha: float
    """Interdomain coupling [dimensionless]"""

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


JA_SCALAR_COEFFS_INITIAL = JAScalarCoeffs(
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
    # TODO: split into a list for ascending/descending H
    H_M_B: npt.NDArray[np.float64]
    """
    An nx3 vector where the columns are the applied field H, magnetization M, and flux density B, respectively.
    One row per sample point.
    """


# noinspection PyPep8Naming
def ja_solve_scalar(
    coef: JAScalarCoeffs,
    H_0: float = 0,
    M_0: float = 0,
    H_step: float = 0.1,
    direction: int = +1,
) -> Solution:
    """
    Solves the JA model for the given coefficients and initial conditions by sweeping the magnetization in the
    specified direction using the specified step size until saturation, and then retraces the magnetization back
    until saturation in the opposite direction.

    Returns sample points for the H, M, and B fields in the order of their appearance.
    """
    H = float(H_0)
    M = float(M_0)
    H_step = float(H_step)
    if not 0 < H_step or not np.isfinite(H_step):
        raise ValueError(f"H_step invalid: {H_step}")
    direction = int(direction)
    if direction not in {-1, +1}:
        raise ValueError(f"direction invalid: {direction}")

    HMB: list[tuple[float, float, float]] = []

    # TODO: Implement

    return Solution(H_M_B=np.array(HMB, dtype=np.float64))


# noinspection PyPep8Naming
def ja_dM_dH(coef: JAScalarCoeffs, *, H: float, M: float, dH_diff_step: float) -> float:
    """
    Returns dM/dH for the 1D Jiles–Atherton model, given the current applied field and magnetization.

    >>> fun = lambda H, M: ja_dM_dH(JA_SCALAR_COEFFS_INITIAL, H=H, M=M, dH_diff_step=1e-3)
    >>> assert np.isclose(fun(-1, 0), fun(+1, 0))
    >>> assert np.isclose(fun(-1, 0.8e6), fun(+1, -0.8e6))
    >>> assert np.isclose(fun(+1, 0.8e6), fun(-1, -0.8e6))
    """
    H_e = H + coef.alpha * M
    M_an = coef.M_s * langevin(np.abs(H_e) / coef.a) * np.sign(H_e)
    # Derivative of anhysteretic magnetization M_an wrt effective H-field H_e.
    # Derived from the above with simplification: langevin(abs(H_e)/a)*sign(H_e) => langevin(H_e/a).
    dM_an_dH_e = coef.M_s / coef.a * dL_dx(H_e / coef.a)

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


def main() -> None:
    basicConfig(level="DEBUG", format="%(asctime)s %(levelname)-3.3s %(name)s: %(message)s")

    coef = copy.copy(JA_SCALAR_COEFFS_INITIAL)

    print(ja_dM_dH(coef, H=-1, M=0, dH_diff_step=1e-3))
    print(ja_dM_dH(coef, H=+0, M=0, dH_diff_step=1e-3))
    print(ja_dM_dH(coef, H=+1, M=0, dH_diff_step=1e-3))

    print(ja_dM_dH(coef, H=-1, M=0.8e6, dH_diff_step=1e-3))
    print(ja_dM_dH(coef, H=+0, M=0.8e6, dH_diff_step=1e-3))
    print(ja_dM_dH(coef, H=+1, M=0.8e6, dH_diff_step=1e-3))

    print(ja_dM_dH(coef, H=-1, M=-0.8e6, dH_diff_step=1e-3))
    print(ja_dM_dH(coef, H=+0, M=-0.8e6, dH_diff_step=1e-3))
    print(ja_dM_dH(coef, H=+1, M=-0.8e6, dH_diff_step=1e-3))


def langevin(x: float) -> float:
    """
    L(x) = coth(x) - 1/x
    For tensors, the function is applied element-wise.
    """
    if np.abs(x) < EPSILON:  # For very small |x|, use the series expansion ~ x/3
        return x / 3.0
    return float(1.0 / np.tanh(x) - 1.0 / x)


# noinspection PyPep8Naming
def dL_dx(x: float) -> float:
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


if __name__ == "__main__":
    main()
