# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

"""
Optimization utilities.
"""

import time
import warnings
import dataclasses
from typing import Callable
from logging import getLogger
import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
from . import ja, bh
from .util import njit


def loss_demag_loop_key_points(bh_ref: npt.NDArray[np.float64], sol: ja.Solution) -> float:
    """
    Dissimilarity metric between the BH curve and the JA model prediction that considers only H_c, B_r, and BH_max;
    all of these parameters must be positive (i.e., the curve must pass through the second quadrant).

    The loss is normalized against the reference parameter values --- the loss dimension is therefore unity;
    this means that loss values below about 1e-3 indicate a decent fit.
    Normalization is essential because H and B can differ by orders of magnitude; without normalization,
    the optimizer would focus on one dimension disproportionately at the expense of the other.

    This metric is only suitable for global optimization when no good priors are available.
    For fine-tuning, use other loss functions that consider the shape of the curve instead of just the key points.
    """
    bh_sol = np.delete(sol.HMB_major_descending[::-1], 1, axis=1)  # Drop M values
    assert np.all(np.diff(bh_sol[:, 0]) > 0)
    if not bh_sol[0, 0] < 0 <= bh_sol[-1, 0]:
        _logger.info("Solution does not include H=0; assume infinite loss: %s", bh_sol[:, 0].tolist())
        return np.inf
    ref_H_c, ref_B_r, ref_BH_max = bh.extract_H_c_B_r_BH_max_from_major_descending_loop(bh_ref)
    sol_H_c, sol_B_r, sol_BH_max = bh.extract_H_c_B_r_BH_max_from_major_descending_loop(bh_sol)
    eps = 1e-6
    loss_H_c = np.abs(ref_H_c - sol_H_c) / max(abs(ref_H_c), eps)
    loss_B_r = np.abs(ref_B_r - sol_B_r) / max(abs(ref_B_r), eps)
    loss_BH_max = np.abs(ref_BH_max - sol_BH_max) / max(abs(ref_BH_max), eps)
    return float(loss_H_c + loss_B_r + loss_BH_max)


def loss_shape(bh_ref: npt.NDArray[np.float64], sol: ja.Solution) -> float:
    """
    Dissimilarity metric between the BH curve and the JA model prediction that considers the shape of the curve.

    Note that this is not the same as simply comparing the B values at specific H points through interpolation;
    that approach creates a very steep and narrow wedge in the loss hyperplane because the function becomes
    extremely sensitive to the horizontal displacement of the knee. As a result, the minimum is hard to find
    for the optimizer, which causes poor convergence.

    Normalization is done by mapping B/mu_0=M+H such that both axes have comparable magnitudes (both in ampere/meter).
    The loss dimension is therefore ampere/meter, which can be relied on to choose a good convergence threshold.
    Normalization is essential because H and B can differ by orders of magnitude; without normalization,
    the optimizer would focus on one dimension disproportionately at the expense of the other.
    """
    bh_ref = np.copy(bh_ref)
    bh_ref[:, 1] /= ja.mu_0
    bh_sol = np.delete(sol.HMB_major_descending, 1, axis=1)  # Drop M values
    bh_sol[:, 1] /= ja.mu_0
    return float(_rms_distance_points_to_polyline(bh_ref, bh_sol))


@njit(nogil=True)
def _rms_distance_points_to_polyline(points: npt.NDArray[np.float64], polyline: npt.NDArray[np.float64]) -> np.float64:
    """
    >>> fun = lambda r, s: float(_rms_distance_points_to_polyline(np.array(r), np.array(s)))
    >>> fun([(0,0),(1,1),(2,2)], [(0,0),(1,1),(2,2)])
    0.0
    >>> round(fun([(0,0),(1,1),(2,2)], [(0,0),(1,1),(3,1)]), 3)
    0.577
    """
    assert len(points.shape) == 2 and points.shape[1] == 2
    assert len(polyline.shape) == 2 and polyline.shape[1] == 2
    d = np.array([_squared_distance_point_to_polyline(q, polyline) for q in points], dtype=np.float64)
    return np.sqrt(np.mean(d))  # type: ignore


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
        return np.linalg.norm(point - polyline[0])

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


def loss_naive(bh_ref: npt.NDArray[np.float64], sol: ja.Solution) -> float:
    """
    Dissimilarity metric between the BH curve and the JA model prediction that compares B values for each H
    in the reference curve through interpolation.
    Provided for completeness.
    """
    bh_sol = np.delete(sol.HMB_major_descending[::-1], 1, axis=1)  # Drop M values
    assert np.all(np.diff(bh_sol[:, 0]) > 0)
    if not bh_sol[0, 0] < 0 <= bh_sol[-1, 0]:
        _logger.info("Solution does not include H=0; assume infinite shape loss: %s", bh_sol[:, 0].tolist())
        return np.inf

    # Trim bh_ref such that it matches the range of H in bh_sol (usually this is a no-op).
    bh_ref = bh_ref[(bh_ref[:, 0] >= bh_sol[0, 0]) & (bh_ref[:, 0] <= bh_sol[-1, 0])]
    assert bh_ref[0, 0] >= bh_sol[0, 0] and bh_ref[-1, 0] <= bh_sol[-1, 0]

    # Interpolate the solution to match the reference curve and calculate the loss.
    # TODO: this is not enough; both should be interpolated and resampled on a regular fine lattice;
    # otherwise, the loss is mainly driven by the regions with high sample density.
    bh_sol_interp = np.interp(bh_ref[:, 0], bh_sol[:, 0], bh_sol[:, 1])
    return float(np.mean(np.abs(bh_ref[:, 1] - bh_sol_interp)))


@dataclasses.dataclass(frozen=True)
class ObjectiveFunctionResult:
    loss: float
    done: bool
    """If true, the optimization should be stopped, the result is considered good enough."""


ObjectiveFunction = Callable[[ja.Coef], ObjectiveFunctionResult]


def make_objective_function(
    bh_ref: npt.NDArray[np.float64],
    loss_fun: Callable[[npt.NDArray[np.float64], ja.Solution], float],
    *,
    tolerance: float,
    H_range_max: float,
    stop_loss: float = -np.inf,
    stop_evals: int = 10**10,
    cb_on_best: Callable[[int, float, ja.Coef, ja.Solution], None] | None = None,
) -> ObjectiveFunction:
    H_stop_range = float(np.max(np.abs(bh_ref[0, :]))), float(H_range_max)
    epoch = 0
    best_loss = np.inf

    def obj_fn(c: ja.Coef) -> ObjectiveFunctionResult:
        nonlocal epoch, best_loss
        sol: ja.Solution | None = None
        started_at = time.monotonic()
        try:
            sol = ja.solve(c, tolerance=tolerance, H_stop_range=H_stop_range, no_ascent=True)
            loss = loss_fun(bh_ref, sol)
        except ja.SolverError as ex:
            _logger.debug("Solver error: %s: %s", type(ex).__name__, ex)
            loss = np.inf
        elapsed = time.monotonic() - started_at
        is_best = loss < best_loss
        best_loss = loss if is_best else best_loss
        (_logger.info if is_best else _logger.debug)(
            "Solution #%s: %s loss=%.6f, H_stop_range=%s, tolerance=%f, elapsed=%.1fms, n_points=%s",
            epoch,
            c,
            loss,
            H_stop_range,
            tolerance,
            elapsed * 1e3,
            len(sol.HMB_major_descending) if sol else None,
        )
        if is_best and cb_on_best is not None and sol:
            cb_on_best(epoch, loss, c, sol)
        epoch += 1
        done = loss < stop_loss or epoch >= stop_evals
        return ObjectiveFunctionResult(loss=loss, done=done)

    return obj_fn


def fit_global(
    x_0: ja.Coef,
    x_min: ja.Coef,
    x_max: ja.Coef,
    obj_fn: ObjectiveFunction,
    *,
    tolerance: float | None = None,
) -> ja.Coef:
    # noinspection PyTypeChecker
    v_0, v_min, v_max = (np.array(dataclasses.astuple(j)) for j in (x_0, x_min, x_max))
    is_done = False

    def obj_fn_proxy(x: npt.NDArray[np.float64]) -> float:
        nonlocal is_done
        x = np.minimum(np.maximum(x, v_min), v_max)  # Some optimizers may violate the bounds
        ofr = obj_fn(ja.Coef(*map(float, x)))
        is_done = ofr.done
        return ofr.loss

    _logger.info("Global optimization: x_0=%s, x_min=%s, x_max=%s", x_0, x_min, x_max)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = opt.differential_evolution(
            obj_fn_proxy,
            [(lo, hi) for lo, hi in zip(v_min, v_max)],
            x0=v_0,
            maxiter=10**4,
            tol=tolerance or 0.01,
            callback=lambda *_, **_k: is_done,
        )
    _logger.info("Global optimization result:\n%s", res)
    # We have to check is_done because an early stop is considered an error by the optimizer (strange but true).
    if res.success or (is_done and np.all(np.isfinite(res.x))):
        return ja.Coef(*map(float, res.x))
    raise RuntimeError(f"Global optimization failed: {res.message}")


def fit_local(
    x_0: ja.Coef,
    x_min: ja.Coef,
    x_max: ja.Coef,
    obj_fn: ObjectiveFunction,
    *,
    use_gradient: bool = False,
) -> ja.Coef:
    # noinspection PyTypeChecker
    v_0, v_min, v_max = (np.array(dataclasses.astuple(j)) for j in (x_0, x_min, x_max))
    is_done = False

    def fun(x: npt.NDArray[np.float64]) -> float:
        nonlocal is_done
        assert np.isfinite(x).all(), x
        ofr = obj_fn(ja.Coef(*map(float, x)))
        is_done = ofr.done
        return ofr.loss if np.isfinite(ofr.loss) else 1e100

    def cb(intermediate_result: opt.OptimizeResult) -> None:
        _ = intermediate_result
        if is_done:
            raise StopIteration

    maxiter = 10**5
    bounds = [(lo, hi) for lo, hi in zip(v_min, v_max)]

    if use_gradient:
        # The default finite differences step size used by L-BFGS-B is too small for this problem.
        # Since this is the final local optimization, we do not expect the arguments to stray far from x_0.
        diff_eps = np.maximum(np.abs(v_0) * 1e-6, 1e-9)
        tol = 1e-15  # With the default tolerance, the optimizer tends to stop prematurely.
        _logger.info("Gradient-based local optimization: x_0=%s, diff_eps=%s", x_0, diff_eps)
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html; "tol" sets "ftol" but not "gtol"
        res = opt.minimize(
            fun, v_0, bounds=bounds, callback=cb, tol=tol, options={"eps": diff_eps, "gtol": tol, "maxiter": maxiter}
        )
    else:
        _logger.info("Gradient-free local optimization: x_0=%s", x_0)
        tol = 1e-15
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html; "tol" sets "fatol" and "xatol"
        res = opt.minimize(
            fun, v_0, method="Nelder-Mead", bounds=bounds, callback=cb, tol=tol, options={"maxiter": maxiter}
        )

    _logger.info("Local optimization result:\n%s", res)
    if res.success or (is_done and np.all(np.isfinite(res.x))):
        return ja.Coef(*map(float, res.x))
    raise RuntimeError(f"Local optimization failed: {res.message}")


_logger = getLogger(__name__)
