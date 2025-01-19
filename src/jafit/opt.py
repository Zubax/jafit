# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

"""
Optimization utilities.
"""

import time
import warnings
import dataclasses
from typing import Callable, Any
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
    H_stop_range: tuple[float, float],
    callback: Callable[[int, Coef, tuple[Solution, float] | Exception], None],
    decimate_solution_to: int = 10_000,
    stop_loss: float = -np.inf,
    stop_evals: int = 10**10,
    solver_extra_args: dict[str, Any] | None = None,
) -> ObjectiveFunction:
    """
    WARNING: cb_on_best() may be invoked from a different thread concurrently!.
    """
    g_epoch = 0
    g_best_loss = np.inf

    def obj_fn(c: Coef) -> ObjectiveFunctionResult:
        nonlocal g_epoch, g_best_loss
        this_epoch = g_epoch
        g_epoch += 1

        sol: Solution | None = None
        started_at = time.monotonic()
        elapsed_loss = 0.0
        try:
            sol = solve(c, H_stop_range=H_stop_range, **(solver_extra_args or {}))
        except SolverError as ex:
            callback(this_epoch, c, ex)
            error = f"{type(ex).__name__}: {ex}"
            loss = np.inf
        else:
            error = ""
            loss_started_at = time.monotonic()
            loop = HysteresisLoop(descending=sol.descending, ascending=sol.ascending)
            loss = loss_fun(ref, loop.decimate(decimate_solution_to))
            elapsed_loss = time.monotonic() - loss_started_at
        elapsed = time.monotonic() - started_at

        is_best = loss < g_best_loss
        g_best_loss = loss if is_best else g_best_loss

        if error:
            _logger.warning("#%05d ❌ %6.3fs: %s %s", this_epoch, elapsed, c, error)
        else:
            _logger.info(
                "#%05d %s %6.3fs: %s loss=%.6f t_loss=%.3f pts=%d",
                this_epoch,
                "🔵💚"[is_best],
                elapsed,
                c,
                loss,
                elapsed_loss,
                (len(sol.descending) + len(sol.ascending)) if sol else 0,
            )
        if is_best and sol:
            callback(this_epoch, c, (sol, loss))
        return ObjectiveFunctionResult(loss=loss, done=loss < stop_loss or this_epoch >= stop_evals)

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
        if ofr.done:
            is_done = True
        return ofr.loss

    def cb(intermediate_result: opt.OptimizeResult) -> None:
        _ = intermediate_result
        if is_done:
            raise StopIteration

    _logger.info("Global optimization: x_0=%s, x_min=%s, x_max=%s", x_0, x_min, x_max)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = opt.differential_evolution(
            obj_fn_proxy,
            [(lo, hi) for lo, hi in zip(v_min, v_max)],
            x0=v_0,
            mutation=(0.5, 1.0),
            recombination=0.7,
            popsize=15,
            maxiter=10**6,
            tol=tolerance or 0.01,
            callback=cb,
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
        tol = 1e-12  # With the default tolerance, the optimizer tends to stop prematurely.
        _logger.info("Gradient-based local optimization: x_0=%s, diff_eps=%s", x_0, diff_eps)
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html; "tol" sets "ftol" but not "gtol"
        res = opt.minimize(
            fun, v_0, bounds=bounds, callback=cb, tol=tol, options={"eps": diff_eps, "gtol": tol, "maxiter": maxiter}
        )
    else:
        _logger.info("Gradient-free local optimization: x_0=%s", x_0)
        tol = 1e-12
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html; "tol" sets "fatol" and "xatol"
        res = opt.minimize(
            fun, v_0, method="Nelder-Mead", bounds=bounds, callback=cb, tol=tol, options={"maxiter": maxiter}
        )

    _logger.info("Local optimization result:\n%s", res)
    if res.success or (is_done and np.all(np.isfinite(res.x))):
        return Coef(*map(float, res.x))
    raise RuntimeError(f"Local optimization failed: {res.message}")


_logger = getLogger(__name__)
