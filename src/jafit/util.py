# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

from typing import Any, Callable
import numpy as np
import numpy.typing as npt
import scipy.interpolate

try:
    from numba import jit, njit
except ImportError:

    def jit(**_: Any) -> Callable[[Callable], Callable]:
        return lambda f: f

    njit = jit


def interpolate(at: npt.NDArray[np.float64], points_xy: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Interpolates the given points at the specified locations using the PCHIP method
    (piecewise monotonic cubic non-overshooting).

    >>> interpolate(np.array([0.0, 0.5, 1.0, 1.5, 2.0]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])).tolist()
    [0.0, 0.75, 1.0, 0.75, 0.0]
    """
    # noinspection PyUnresolvedReferences
    return scipy.interpolate.PchipInterpolator(points_xy[:, 0], points_xy[:, 1])(at)
