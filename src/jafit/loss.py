# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

from typing import Callable
from logging import getLogger
import numpy as np
import numpy.typing as npt
from .mag import extract_H_c_B_r_BH_max, HysteresisLoop
from .util import njit, interpolate

LossFunction = Callable[[HysteresisLoop], float]


def make_demag_key_points(ref: HysteresisLoop) -> LossFunction:
    """
    Dissimilarity metric that considers only H_c, B_r, and BH_max; all of these parameters should be much greater than 0
    (i.e., the H(M) curve should pass through the second quadrant far from the origin).
    Both the reference loop and the solution loop must have their descending branches defined.

    The loss is normalized against the reference parameter values --- the loss dimension is therefore unity;
    this means that loss values below about 1e-3 indicate a decent fit.
    Normalization is essential to avoid the optimizer focusing on one metric disproportionately
    at the expense of the others.

    This metric is only suitable for global optimization when no good priors are available.
    For finer optimization, use other loss functions that consider the shape of the curve instead of just the key points.
    """
    if len(ref.descending) == 0:
        raise ValueError("Descending curve missing")
    ref_H_c, ref_B_r, ref_BH_max = extract_H_c_B_r_BH_max(ref.descending)

    def fun(sol: HysteresisLoop, *, eps: float = 1e-9) -> float:
        if not sol.descending[0, 0] < 0 <= sol.descending[-1, 0]:
            _logger.info("Solution does not include H=0; assume infinite loss: %s", sol.descending[:, 0].tolist())
            return np.inf
        if len(sol.descending) == 0:
            raise ValueError("Descending curve missing")
        sol_H_c, sol_B_r, sol_BH_max = extract_H_c_B_r_BH_max(sol.descending)
        loss_H_c = np.abs(ref_H_c - sol_H_c) / max(abs(ref_H_c), eps)
        loss_B_r = np.abs(ref_B_r - sol_B_r) / max(abs(ref_B_r), eps)
        loss_BH_max = np.abs(ref_BH_max - sol_BH_max) / max(abs(ref_BH_max), eps)
        return float(loss_H_c + loss_B_r + loss_BH_max)

    return fun


def make_magnetization(ref: HysteresisLoop, *, lattice_size: int = 10**4) -> LossFunction:
    """
    The ordinary dissimilarity metric that computes sqrt(sum(( M_ref(H)-M_sol(H) )**2)/n)
    for every H in the regular lattice of the specified size n on every branch.
    Both curves are interpolated using the PCHIP method (piecewise monotonic cubic non-overshooting).
    The computed loss values per loop branch are averaged.

    Normalization is not done because both coordinates are in the same units [A/m];
    this is also the dimension of the computed loss value.
    """
    H_lattice = np.linspace(*ref.H_range, lattice_size)

    def loss(hm_ref: npt.NDArray[np.float64], hm_sol: npt.NDArray[np.float64]) -> float:
        M_ref = interpolate(H_lattice, hm_ref, extrapolate=True)  # type: ignore
        M_sol = interpolate(H_lattice, hm_sol, extrapolate=True)  # type: ignore
        return float(np.sqrt(np.mean((M_ref - M_sol) ** 2)))

    def fun(sol: HysteresisLoop) -> float:
        loss_values = []
        if len(ref.descending) and len(sol.descending):
            loss_values.append(loss(ref.descending, sol.descending))
        if len(ref.ascending) and len(sol.ascending):
            loss_values.append(loss(ref.ascending, sol.ascending))
        if not loss_values:
            raise ValueError("No same-side hysteresis branches to compare")
        return float(np.mean(loss_values))

    return fun


def make_nearest(ref: HysteresisLoop, *, priority_region_error_gain: float = 1.0) -> LossFunction:
    """
    Dissimilarity metric that computes the mean distance between each point of the reference H(M) curves and the
    nearest point on the solution H(M) curves. The computed loss values per loop branch are averaged.

    The points in the reference curve should be spaced more or less uniformly; otherwise, the loss will be
    dominated by the regions with higher point density -- although, sometimes it is the desired behavior.
    One way to ensure uniform sampling is to perform spline interpolation using ``interpolate_spline_equidistant()``
    before calling this function. The number of sample points should usually be somewhere between 100..1000,
    depending on the complexity of the shape.

    The data on both axes is normalized such that the reference values are in the range [-1, +1],
    and the solution values are scaled accordingly. The computed loss value is therefore also normalized.
    The normalization is done to ensure that the optimizer assigns comparable importance to all dimensions.

    The priority_region_error_gain is used to assign higher weights to quadrants 1 and 2 of the descending branch
    and quadrants 3 and 4 of the ascending branch: the error in the priority region is multiplied by this value
    and kept as-is in the other regions. Set to 1 to disable this feature.

    The computational complexity is high, not recommended for large datasets without prior downsampling.
    """
    priority_region_error_gain = float(priority_region_error_gain)
    if not np.isfinite(priority_region_error_gain) or priority_region_error_gain < 1e-9:
        raise ValueError(f"Invalid {priority_region_error_gain=}")

    def abs_max(m: npt.NDArray[np.float64], col: int) -> float:
        return float(np.abs(m[:, col]).max(initial=0.0))

    H_scale = max(abs_max(ref.descending, 0), abs_max(ref.ascending, 0), 1.0)
    M_scale = max(abs_max(ref.descending, 1), abs_max(ref.ascending, 1), 1.0)

    def scale(m: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        m = m.copy()
        m[:, 0] /= H_scale
        m[:, 1] /= M_scale
        return m

    preg = priority_region_error_gain
    rd, ra = scale(ref.descending), scale(ref.ascending)
    _logger.info("Normalized H scale: %f, M scale: %f; PREG=%f", H_scale, M_scale, priority_region_error_gain)

    def fun(sol: HysteresisLoop) -> float:
        sd, sa = scale(sol.descending), scale(sol.ascending)
        loss: list[np.float64] = []

        if len(rd) and len(sd):
            d = np.array([_distance_point_to_polyline(q, sd) for q in rd])
            w = np.array([1.0 if m < 0 else preg for _h, m in rd])
            loss.append(np.mean(d * w))

        if len(ra) and len(sa):
            d = np.array([_distance_point_to_polyline(q, sa) for q in ra])
            w = np.array([1.0 if m > 0 else preg for _h, m in ra])
            loss.append(np.mean(d * w))

        if not loss:
            raise ValueError("No same-side hysteresis branches to compare")
        return float(np.mean(loss))

    return fun


@njit
def _distance_point_to_polyline(point: npt.NDArray[np.float64], polyline: npt.NDArray[np.float64]) -> np.float64:
    """
    See ``_squared_distance_point_to_polyline()`` for the description.

    >>> fun = lambda p, line: float(_distance_point_to_polyline(np.array(p), np.array(line)))
    >>> fun((0,0), [(-1,1),(1,1),(2,2)])
    1.0
    >>> round(fun((2,1), [(-1,1),(1,1),(2,2)]), 3)
    0.707
    >>> round(fun((1,2), [(-1,1),(1,1),(2,2)]), 3)
    0.707
    >>> fun((2,2), [(-1,1),(1,1),(2,2)])
    0.0
    >>> with np.errstate(divide="ignore", invalid="ignore"):
    ...     fun((0,0), [(-1,1),(1,1),(1,1),(2,2)])
    1.0
    >>> with np.errstate(divide="ignore", invalid="ignore"):
    ...     fun((2,2), [(-1,1),(1,1),(1,1),(2,2)])
    0.0
    """
    return np.sqrt(_squared_distance_point_to_polyline(point, polyline))  # type: ignore


@njit(nogil=True)
def _squared_distance_point_to_polyline(
    point: npt.NDArray[np.float64],
    polyline: npt.NDArray[np.float64],
) -> np.float64:
    """
    Computes the minimum distance (squared) from the given point to a piecewise-linear curve.
    The point is of shape (2,) and the polyline is of shape (N, 2).
    Returns the minimal distance from the point to any point on the curve.
    Each polyline segment should have a non-zero length; if this is not the case, wrap the call with this
    (this context manager cannot be used internally here because Numba doesn't support it in nopython mode)::

        with np.errstate(divide="ignore", invalid="ignore"):
            ...

    >>> fun = lambda p, line: float(np.sqrt(_squared_distance_point_to_polyline(np.array(p), np.array(line))))
    >>> fun((0,0), [(-1,1),(1,1),(2,2)])
    1.0
    >>> round(fun((2,1), [(-1,1),(1,1),(2,2)]), 3)
    0.707
    >>> round(fun((1,2), [(-1,1),(1,1),(2,2)]), 3)
    0.707
    >>> fun((2,2), [(-1,1),(1,1),(2,2)])
    0.0
    >>> with np.errstate(divide="ignore", invalid="ignore"):
    ...     fun((0,0), [(-1,1),(1,1),(1,1),(2,2)])
    1.0
    >>> with np.errstate(divide="ignore", invalid="ignore"):
    ...     fun((2,2), [(-1,1),(1,1),(1,1),(2,2)])
    0.0
    """
    assert point.shape == (2,)
    assert len(polyline.shape) == 2 and polyline.shape[1] == 2
    if len(polyline) < 1:
        return np.float64(np.nan)
    if len(polyline) < 2:
        return np.linalg.norm(point - polyline[0])  # type: ignore

    # Let P[i] = polyline[i], and d[i] = P[i+1] - P[i]. We'll operate over all segments in a vectorized manner.
    P = polyline[:-1]  # shape = (N-1, 2)
    P_next = polyline[1:]  # shape = (N-1, 2)

    # Vector for each segment: d[i] = P[i+1] - P[i]
    d = P_next - P  # shape = (N-1, 2)

    # Vector from each P[i] to the point: point - P[i]. Broadcast point (shape (2,)) against P (shape (N-1, 2))
    PQ = point - P  # shape = (N-1, 2)

    # Length squared of each segment d[i], used for normalization.
    # We assume that no degenerate segments are present (i.e., d[i] != 0 for all i).
    # d_len_sq = np.einsum("ij,ij->i", d, d)  # shape = (N-1,)
    d_len_sq = d[:, 0] * d[:, 0] + d[:, 1] * d[:, 1]  # shape = (N-1,)

    # Dot product of PQ and d along axis=1
    # dot_prod = np.einsum("ij,ij->i", PQ, d)  # shape = (N-1,)
    dot_prod = PQ[:, 0] * d[:, 0] + PQ[:, 1] * d[:, 1]  # shape = (N-1,)

    # Parameter t of the projection of the point onto the infinite line: t = (PQ . d) / |d|^2
    # Then we clamp in [0,1] to ensure the closest point is within the segment.
    t = np.clip(np.nan_to_num(dot_prod / d_len_sq), 0, 1)

    # The closest point on each segment in parametric form: P[i] + t[i] * d[i]
    closest_points = P + d * t[:, None]

    # Distances squared from the point to each of these closest points.
    diff = point - closest_points  # shape (N, 2)
    # dist_sq = np.einsum("ij,ij->i", diff, diff)
    dist_sq = diff[:, 0] * diff[:, 0] + diff[:, 1] * diff[:, 1]
    return dist_sq.min()  # type: ignore


_logger = getLogger(__name__)
