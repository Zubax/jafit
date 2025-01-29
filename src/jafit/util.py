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


def interpolate(
    at: npt.NDArray[np.float64], points_xy: npt.NDArray[np.float64], extrapolate: bool = False
) -> npt.NDArray[np.float64]:
    """
    Interpolates the given points at the specified locations using the PCHIP method
    (piecewise monotonic cubic non-overshooting).

    >>> interpolate(np.array([0.0, 0.5, 1.0, 1.5, 2.0]), np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 0.0]])).tolist()
    [0.0, 0.75, 1.0, 0.75, 0.0]
    """
    # noinspection PyUnresolvedReferences
    return scipy.interpolate.PchipInterpolator(  # type: ignore
        points_xy[:, 0],
        points_xy[:, 1],
        extrapolate=extrapolate,
    )(at)


def interpolate_spline_equidistant(
    n_samples: int,
    points_xy: npt.NDArray[np.float64],
    *,
    spline_degree: int = 1,
) -> npt.NDArray[np.float64]:
    """
    Interpolates the given points at the specified number of points spaced an equal distance between each other.

    >>> curve = np.array([(0,0),(1,1),(2,4),(3,9),])
    >>> interpolant = interpolate_spline_equidistant(5, curve)
    >>> interpolant.round(2).tolist()
    [[0.0, 0.0], [1.32, 1.95], [2.05, 4.26], [2.53, 6.63], [3.0, 9.0]]
    >>> np.sqrt((np.diff(interpolant, axis=0)**2).sum(axis=1)).round(1).tolist()
    [2.4, 2.4, 2.4, 2.4]

    >>> interpolate_spline_equidistant(5, np.array([])).tolist()
    []
    """
    if spline_degree < 1:
        raise ValueError(f"Invalid spline degree: {spline_degree}")
    if len(points_xy) == 0 or n_samples < 1:
        return np.empty((0, 2), dtype=np.float64)
    # noinspection PyUnresolvedReferences
    spline, _ = scipy.interpolate.make_splprep(points_xy.T, k=spline_degree)
    lattice = np.linspace(0.0, 1.0, n_samples)
    return spline(lattice).T  # type: ignore


def relative_distance(x: npt.NDArray[np.float64], y: npt.NDArray[np.float64], eps: float = 1e-9) -> float:
    """
    Computes the Euclidean distance between x and y normalized by the larger norm of x and y.
    Can be used to assess how close two points are to each other in a relative sense.
    Zero means that the points are at the same location.

    >>> relative_distance(np.array([9, 0]), np.array([10, 0]))
    0.1
    >>> relative_distance(np.array([0, 0]), np.array([10, 0]))
    1.0
    >>> relative_distance(np.array([0, 0]), np.array([0, 0]))
    0.0
    """
    dist = np.linalg.norm(x - y)
    denom = max(np.linalg.norm(x), np.linalg.norm(y))  # type: ignore
    if denom < eps:
        return 0.0 if dist < eps else np.inf
    return float(dist / denom)
