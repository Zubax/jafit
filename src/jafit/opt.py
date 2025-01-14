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
from .ja import Solution, Coef, solve, SolverError
from .mag import HysteresisLoop


@dataclasses.dataclass(frozen=True)
class ObjectiveFunctionResult:
    loss: float
    done: bool
    """If true, the optimization should be stopped, the result is considered good enough."""


ObjectiveFunction = Callable[[Coef], ObjectiveFunctionResult]

LossFunction = Callable[[HysteresisLoop, HysteresisLoop], float]
"""
The first is the reference, the second is the evaluated solution.
"""


def make_objective_function(
    ref: HysteresisLoop,
    loss_fun: LossFunction,
    *,
    tolerance: float,
    H_range_max: float,
    stop_loss: float = -np.inf,
    stop_evals: int = 10**10,
    cb_on_best: Callable[[int, float, Coef, Solution], None] | None = None,
) -> ObjectiveFunction:
    H_stop_range = float(np.max(np.abs(ref.H_range))), float(H_range_max)
    epoch = 0
    best_loss = np.inf

    def obj_fn(c: Coef) -> ObjectiveFunctionResult:
        nonlocal epoch, best_loss
        sol: Solution | None = None
        started_at = time.monotonic()
        try:
            sol = solve(c, tolerance=tolerance, H_stop_range=H_stop_range)
            loss = loss_fun(ref, sol.major_loop)
        except SolverError as ex:
            _logger.debug("Solver error: %s: %s", type(ex).__name__, ex)
            loss = np.inf
        elapsed = time.monotonic() - started_at
        is_best = loss < best_loss
        best_loss = loss if is_best else best_loss
        (_logger.info if is_best else _logger.debug)(
            "Solution #%s: %s loss=%.6f, H_stop_range=%s, tolerance=%f, elapsed=%.1fms",
            epoch,
            c,
            loss,
            H_stop_range,
            tolerance,
            elapsed * 1e3,
        )
        if is_best and cb_on_best is not None and sol:
            cb_on_best(epoch, loss, c, sol)
        epoch += 1
        done = loss < stop_loss or epoch >= stop_evals
        return ObjectiveFunctionResult(loss=loss, done=done)

    return obj_fn


def fit_global(
    x_0: Coef,
    x_min: Coef,
    x_max: Coef,
    obj_fn: ObjectiveFunction,
    *,
    tolerance: float | None = None,
) -> Coef:
    # noinspection PyTypeChecker
    v_0, v_min, v_max = (np.array(dataclasses.astuple(j)) for j in (x_0, x_min, x_max))
    is_done = False

    def obj_fn_proxy(x: npt.NDArray[np.float64]) -> float:
        nonlocal is_done
        x = np.minimum(np.maximum(x, v_min), v_max)  # Some optimizers may violate the bounds
        ofr = obj_fn(Coef(*map(float, x)))
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
        return Coef(*map(float, res.x))
    raise RuntimeError(f"Global optimization failed: {res.message}")


def fit_local(
    x_0: Coef,
    x_min: Coef,
    x_max: Coef,
    obj_fn: ObjectiveFunction,
    *,
    use_gradient: bool = False,
) -> Coef:
    # noinspection PyTypeChecker
    v_0, v_min, v_max = (np.array(dataclasses.astuple(j)) for j in (x_0, x_min, x_max))
    is_done = False

    def fun(x: npt.NDArray[np.float64]) -> float:
        nonlocal is_done
        assert np.isfinite(x).all(), x
        ofr = obj_fn(Coef(*map(float, x)))
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
        return Coef(*map(float, res.x))
    raise RuntimeError(f"Local optimization failed: {res.message}")


_logger = getLogger(__name__)
