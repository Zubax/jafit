# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

"""
Optimization utilities.
"""

import time
import warnings
import dataclasses
from typing import Callable, overload
from logging import getLogger
import numpy as np
import numpy.typing as npt
import scipy.optimize as opt
from .ja import Solution, Coef, SolverError
from .mag import HysteresisLoop


@dataclasses.dataclass(frozen=True)
class ObjectiveFunctionResult:
    loss: float
    done: bool
    """If true, the optimization should be stopped, the result is considered good enough."""


ObjectiveFunction = Callable[[Coef], ObjectiveFunctionResult]

SolveFunction = Callable[[Coef], Solution]

LossFunction = Callable[[HysteresisLoop], float]


def make_objective_function(
    solve_fun: SolveFunction,
    loss_fun: LossFunction,
    *,
    callback: Callable[[int, Coef, tuple[Solution, float] | Exception], None],
    decimate_solution_to: int = 10_000,
    stop_loss: float = -np.inf,
    stop_evals: int = 10**10,
    quiet: bool = False,
) -> ObjectiveFunction:
    """
    WARNING: the callback may be invoked from a different thread concurrently.
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
            sol = solve_fun(c)
        except SolverError as ex:
            callback(this_epoch, c, ex)
            error = f"{type(ex).__name__}: {ex}"
            loss = np.inf
        else:
            error = ""
            loss_started_at = time.monotonic()
            loop = HysteresisLoop(descending=sol.last_descending[::-1], ascending=sol.last_ascending)
            loss = loss_fun(loop.decimate(decimate_solution_to))
            elapsed_loss = time.monotonic() - loss_started_at
        elapsed = time.monotonic() - started_at

        is_best = loss < g_best_loss
        g_best_loss = loss if is_best else g_best_loss

        log_fn = _logger.info if not quiet or is_best else _logger.debug
        if error:
            log_fn("#%05d ❌ %6.3fs: %s %s", this_epoch, elapsed, c, error)
        else:
            log_fn(
                "#%05d %s %6.3fs: %s loss=%.9f t_loss=%.3f pts=%s",
                this_epoch,
                "🔵💚"[is_best],
                elapsed,
                c,
                loss,
                elapsed_loss,
                "+".join(map(str, map(len, sol.branches))) if sol else "~",
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
    conv = CoefVecConverter(x_min, x_max)
    is_done = False

    def obj_fn_proxy(x: npt.NDArray[np.float64]) -> float:
        nonlocal is_done
        x = np.minimum(np.maximum(x, conv(x_min)), conv(x_max))  # Some optimizers violate the bounds
        ofr = obj_fn(conv(x))
        if ofr.done:
            is_done = True
        return ofr.loss

    def cb(intermediate_result: opt.OptimizeResult) -> None:
        _ = intermediate_result
        if is_done:
            raise StopIteration

    _logger.info(
        "Global optimization: x_0=%s, x_min=%s, x_max=%s",
        conv(x_0).tolist(),
        conv(x_min).tolist(),
        conv(x_max).tolist(),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = opt.differential_evolution(
            obj_fn_proxy,
            [(lo, hi) for lo, hi in zip(conv(x_min), conv(x_max))],
            x0=conv(x_0),
            mutation=(0.5, 1.5),
            recombination=0.7,
            popsize=15,
            maxiter=10**6,
            tol=tolerance or 0.005,
            callback=cb,
        )
    _logger.info("Global optimization result:\n%s", res)
    # We have to check is_done because an early stop is considered an error by the optimizer (strange but true).
    if res.success or (is_done and np.all(np.isfinite(res.x))):
        return conv(res.x)  # type: ignore
    raise RuntimeError(f"Global optimization failed: {res.message}")


def fit_local(
    x_0: Coef,
    x_min: Coef,
    x_max: Coef,
    obj_fn: ObjectiveFunction,
    *,
    use_gradient: bool = False,
) -> Coef:
    conv = CoefVecConverter(x_min, x_max)
    is_done = False

    def fun(x: npt.NDArray[np.float64]) -> float:
        nonlocal is_done
        assert np.isfinite(x).all(), x
        ofr = obj_fn(conv(x))
        is_done = ofr.done
        return ofr.loss if np.isfinite(ofr.loss) else 1e100

    def cb(intermediate_result: opt.OptimizeResult) -> None:
        _ = intermediate_result
        if is_done:
            raise StopIteration

    maxiter = 10**5
    bounds = [(lo, hi) for lo, hi in zip(conv(x_min), conv(x_max))]

    if use_gradient:
        # The default finite differences step size used by L-BFGS-B is too small for this problem.
        # Since this is the final local optimization, we do not expect the arguments to stray far from x_0.
        diff_eps = np.maximum(np.abs(conv(x_0)) * 1e-6, 1e-9)
        tol = 1e-12  # With the default tolerance, the optimizer tends to stop prematurely.
        _logger.info("Gradient-based local optimization: x_0=%s, diff_eps=%s", x_0, diff_eps)
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-lbfgsb.html; "tol" sets "ftol" but not "gtol"
        res = opt.minimize(
            fun,
            conv(x_0),
            bounds=bounds,
            callback=cb,
            tol=tol,
            options={"eps": diff_eps, "gtol": tol, "maxiter": maxiter},
        )
    else:
        _logger.info("Gradient-free local optimization: x_0=%s", x_0)
        # https://docs.scipy.org/doc/scipy/reference/optimize.minimize-neldermead.html; "tol" sets "fatol" and "xatol"
        res = opt.minimize(
            fun,
            conv(x_0),
            method="Nelder-Mead",
            bounds=bounds,
            callback=cb,
            options={"maxiter": maxiter, "xatol": 1e-5, "fatol": 1e-4},
        )

    _logger.info("Local optimization result:\n%s", res)
    if res.success or (is_done and np.all(np.isfinite(res.x))):
        return conv(res.x)  # type: ignore
    raise RuntimeError(f"Local optimization failed: {res.message}")


class CoefVecConverter:
    """
    Converts between Coef and a vector.
    If min/max are the same for a dimension, it is omitted from the vector, thus reducing the optimization space.

    Fixed M_s (lo.M_s == hi.M_s):

    >>> lo = Coef(c_r=0.1, M_s=10.0, a=1.0, k_p=2.0, alpha=0.3)
    >>> hi = Coef(c_r=0.9, M_s=10.0, a=3.0, k_p=4.0, alpha=0.7)
    >>> converter = CoefVecConverter(lo, hi)
    >>> coef = Coef(c_r=0.5, M_s=10.0, a=2.0, k_p=3.0, alpha=0.5)
    >>> vec = converter.coef2vec(coef)
    >>> vec.tolist()  # doctest: +ELLIPSIS
    [0.5, 0.5, 0.5, 0.5...]
    >>> new_coef = converter.vec2coef(vec)
    >>> new_coef.c_r, new_coef.M_s, new_coef.a, new_coef.k_p, new_coef.alpha
    (0.5, 10.0, 2.0, 3.0, 0.5)
    >>> converter(coef).tolist()  # doctest: +ELLIPSIS
    [0.5, 0.5, 0.5, 0.5...]
    >>> converter(vec).a
    2.0

    Variable M_s (lo.M_s != hi.M_s):

    >>> lo2 = Coef(c_r=0.1, M_s=10.0, a=1.0, k_p=2.0, alpha=0.3)
    >>> hi2 = Coef(c_r=0.9, M_s=20.0, a=3.0, k_p=4.0, alpha=0.7)
    >>> converter2 = CoefVecConverter(lo2, hi2)
    >>> coef2 = Coef(c_r=0.5, M_s=15.0, a=2.0, k_p=3.0, alpha=0.5)
    >>> vec2 = converter2.coef2vec(coef2)
    >>> vec2.tolist()  # doctest: +ELLIPSIS
    [0.5, 0.5, 0.5, 0.5, 0.5...]
    >>> new_coef2 = converter2.vec2coef(vec2)
    >>> new_coef2.c_r, new_coef2.M_s, new_coef2.a, new_coef2.k_p, new_coef2.alpha
    (0.5, 15.0, 2.0, 3.0, 0.5)
    >>> converter2(coef2).tolist()  # doctest: +ELLIPSIS
    [0.5, 0.5, 0.5, 0.5, 0.5...]
    >>> converter2(vec2).M_s
    15.0

    Same boundary values:

    >>> lo3 = Coef(c_r=0.1, M_s=10.0, a=1.0, k_p=2.0, alpha=0.3)
    >>> hi3 = Coef(c_r=0.9, M_s=20.0, a=1.0, k_p=4.0, alpha=0.3)
    >>> converter3 = CoefVecConverter(lo3, hi3)
    >>> coef3 = Coef(c_r=0.5, M_s=15.0, a=1.0, k_p=3.0, alpha=0.5)
    >>> vec3 = converter3.coef2vec(coef3)
    >>> vec3.tolist()
    [0.5, 0.5, 0.5, 0.5, 0.5]
    >>> new_coef3 = converter3.vec2coef(vec3)
    >>> new_coef3.c_r, new_coef3.M_s, new_coef3.a, new_coef3.k_p, new_coef3.alpha
    (0.5, 15.0, 1.0, 3.0, 0.3)
    """

    def __init__(self, /, lo: Coef, hi: Coef) -> None:
        self._lo = lo
        self._hi = hi
        self._fixed_M_s = np.isclose(lo.M_s, hi.M_s)

    def coef2vec(self, /, c: Coef) -> npt.NDArray[np.float64]:
        if self._fixed_M_s:
            return np.array(
                [
                    self._normalize(c.c_r, self._lo.c_r, self._hi.c_r),
                    self._normalize(c.a, self._lo.a, self._hi.a),
                    self._normalize(c.k_p, self._lo.k_p, self._hi.k_p),
                    self._normalize(c.alpha, self._lo.alpha, self._hi.alpha),
                ],
            )
        return np.array(
            [
                self._normalize(c.c_r, self._lo.c_r, self._hi.c_r),
                self._normalize(c.M_s, self._lo.M_s, self._hi.M_s),
                self._normalize(c.a, self._lo.a, self._hi.a),
                self._normalize(c.k_p, self._lo.k_p, self._hi.k_p),
                self._normalize(c.alpha, self._lo.alpha, self._hi.alpha),
            ],
        )

    def vec2coef(self, /, vec: npt.NDArray[np.float64]) -> Coef:
        v = [float(x) for x in vec]
        if self._fixed_M_s:
            assert len(v) == 4
            return Coef(
                c_r=self._restore(v[0], self._lo.c_r, self._hi.c_r),
                M_s=self._restore(0.5, self._lo.M_s, self._hi.M_s),
                a=self._restore(v[1], self._lo.a, self._hi.a),
                k_p=self._restore(v[2], self._lo.k_p, self._hi.k_p),
                alpha=self._restore(v[3], self._lo.alpha, self._hi.alpha),
            )
        assert len(v) == 5
        return Coef(
            c_r=self._restore(v[0], self._lo.c_r, self._hi.c_r),
            M_s=self._restore(v[1], self._lo.M_s, self._hi.M_s),
            a=self._restore(v[2], self._lo.a, self._hi.a),
            k_p=self._restore(v[3], self._lo.k_p, self._hi.k_p),
            alpha=self._restore(v[4], self._lo.alpha, self._hi.alpha),
        )

    @overload
    def __call__(self, /, x: Coef) -> npt.NDArray[np.float64]: ...
    @overload
    def __call__(self, /, x: npt.NDArray[np.float64]) -> Coef: ...
    def __call__(self, /, x: Coef | npt.NDArray[np.float64]) -> Coef | npt.NDArray[np.float64]:
        if isinstance(x, Coef):
            return self.coef2vec(x)
        if isinstance(x, np.ndarray):
            return self.vec2coef(x)
        raise TypeError(x)

    @staticmethod
    def _normalize(x: float, lo: float, hi: float) -> float:
        if np.isclose(lo, hi):
            return 0.5
        return float((x - lo) / (hi - lo))

    @staticmethod
    def _restore(x: float, lo: float, hi: float) -> float:
        if np.isclose(lo, hi):
            return (lo + hi) * 0.5
        return float(lo + x * (hi - lo))


_logger = getLogger(__name__)
