# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

"""
This is a simple utility that fits the Jiles-Atherton (JA) model coefficients for a given BH curve.
The JA model used here follows that of COMSOL Multiphysics; see the enclosed PDF with the relevant excerpt
from the COMSOL user reference.
"""

from __future__ import annotations

import sys
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar, Iterable
import logging
from pathlib import Path
import numpy as np
import numpy.typing as npt
from . import ja, bh, opt, vis


PLOT_FILE_SUFFIX = ".jafit.png"

BG_EXECUTOR = ThreadPoolExecutor(max_workers=1)
"""We only need one worker."""

T = TypeVar("T")


def make_on_best_callback(
    file_name_prefix: str, bh_ref: npt.NDArray[np.float64]
) -> Callable[[int, float, ja.Coef, ja.Solution], None]:

    def cb(epoch: int, loss: float, coef: ja.Coef, sol: ja.Solution) -> None:
        def bg() -> None:
            plot_file = f"{file_name_prefix}_#{epoch:05}_{loss:.6f}_{coef}{PLOT_FILE_SUFFIX}"
            vis.plot(sol, plot_file, bh_ref)

        # Plotting can take a while, so we do it in a background thread.
        # We do release the GIL in the solver very often, so this is not a problem.
        # Also, future Python versions will eventually have proper support for multithreading.
        BG_EXECUTOR.submit(bg)

    return cb


def do_fit(
    bh_curve: npt.NDArray[np.float64],
    *,
    c_r: float | None,
    M_s: float | None,
    a: float | None,
    k_p: float | None,
    alpha: float | None,
    H_max: float,
    max_evaluations_per_stage: int,
    skip_stages: int,
) -> ja.Coef:
    H_c, B_r, BH_max = bh.extract_H_c_B_r_BH_max_from_major_descending_loop(bh_curve)
    _logger.info(
        "Reference BH curve has %s points. Derived parameters: H_c=%.6f A/m, B_r=%.6f T, BH_max=%.3f J/m^3",
        len(bh_curve),
        H_c,
        B_r,
        BH_max,
    )
    M_s_min = B_r / ja.mu_0  # Heuristic: B=mu_0*(H+M); H=0; B=mu_0*M; hence, M_s>=B_r/mu_0
    _logger.info("Derived minimum M_s: %.6f A/m", M_s_min)

    coef = ja.Coef(
        c_r=_perhaps(c_r, 1e-6),
        M_s=_perhaps(M_s, M_s_min * 1.001),  # Optimizers tend to be unstable if parameters are too close to the bounds
        a=_perhaps(a, 1e4),
        k_p=_perhaps(k_p, 1e4),
        alpha=_perhaps(alpha, 0.1),
    )
    x_min = ja.Coef(c_r=0, M_s=M_s_min, a=0, k_p=0, alpha=0)
    # TODO: better way of setting the upper bounds?
    x_max = ja.Coef(c_r=1, M_s=3e6, a=1e5, k_p=1e5, alpha=1)
    _logger.info("Initial, minimum, and maximum coefficients:\n%s\n%s\n%s", coef, x_min, x_max)

    if (H_c > 1 and B_r > 0.01) and skip_stages < 1:
        _logger.info("Demag knee detected; performing initial H_c|B_r|BH_max optimization")
        coef = opt.fit_global(
            x_0=coef,
            x_min=x_min,
            x_max=x_max,
            obj_fn=opt.make_objective_function(
                bh_curve,
                opt.loss_demag_loop_key_points,
                tolerance=1.0,  # This is a very rough approximation
                H_range_max=H_max,
                stop_loss=1e-3,  # Fine adjustment is meaningless because the solver and the loss fun are crude here
                stop_evals=max_evaluations_per_stage,
                cb_on_best=make_on_best_callback("initial", bh_curve),
            ),
            tolerance=1e-3,
        )
        _logger.info(f"Intermediate result:\n%s", coef)
    else:
        _logger.info("Skipping initial optimization")

    if skip_stages < 2:
        coef = opt.fit_global(
            x_0=coef,
            x_min=x_min,
            x_max=x_max,
            obj_fn=opt.make_objective_function(
                bh_curve,
                opt.loss_shape,
                tolerance=0.01,
                H_range_max=H_max,
                stop_evals=max_evaluations_per_stage,
                cb_on_best=make_on_best_callback("global", bh_curve),
            ),
            tolerance=1e-6,
        )
        _logger.info(f"Intermediate result:\n%s", coef)

    coef = opt.fit_local(
        x_0=coef,
        x_min=x_min,
        x_max=x_max,
        obj_fn=opt.make_objective_function(
            bh_curve,
            opt.loss_shape,
            tolerance=1e-4,
            H_range_max=H_max,
            stop_evals=max_evaluations_per_stage,
            cb_on_best=make_on_best_callback("local", bh_curve),
        ),
    )

    # Emit a warning if the final coefficients are close to the bounds.
    rtol, atol = 1e-6, 1e-9
    # noinspection PyTypeChecker
    for k, v in dataclasses.asdict(coef).items():
        lo, hi = x_min.__getattribute__(k), x_max.__getattribute__(k)
        if np.isclose(v, lo, rtol, atol) or np.isclose(v, hi, rtol, atol):
            _logger.warning("Final %s=%.9f is close to the bounds [%.9f, %.9f]", k, v, lo, hi)

    return coef


def run(
    bh_curve_file: str | None = None,
    *unnamed_args: str,
    c_r: float | None = None,
    M_s: float | None = None,
    a: float | None = None,
    k_p: float | None = None,
    alpha: float | None = None,
    H_max: float = 3e6,
    effort: int = 10**6,
    skip_stages: int = 0,
    **named_args: dict[str, int | float | str],
) -> None:
    if unnamed_args or named_args:
        raise ValueError(f"Unexpected arguments:\nunnamed: {unnamed_args}\nnamed: {named_args}")

    ja_dict = {
        k: (float(v) if v is not None else None) for k, v in dict(c_r=c_r, M_s=M_s, a=a, k_p=k_p, alpha=alpha).items()
    }
    H_max = float(H_max)
    effort = int(effort)

    bh_curve: npt.NDArray[np.float64] | None = None
    if bh_curve_file is not None:
        bh_curve = bh.load(Path(bh_curve_file))
        bh.check(bh_curve)

    if bh_curve is not None:
        _logger.info("Fitting BH curve of %s points with starting parameters: %s", len(bh_curve), ja_dict)
        coef = do_fit(
            bh_curve,
            **ja_dict,
            H_max=H_max,
            max_evaluations_per_stage=effort,
            skip_stages=skip_stages,
        )
        # noinspection PyTypeChecker
        print(*(f"{k}={v}" for k, v in dataclasses.asdict(coef).items()))
    else:
        _logger.info("Fitting will not be performed")
        if any(x is None for x in ja_dict.values()):
            raise ValueError(f"Supplied coefficients are incomplete, and optimization is not requested: {ja_dict}")
        coef = ja.Coef(**ja_dict)  # type: ignore

    _logger.info("Solving and plotting: %s", coef)
    sol = ja.solve(coef, H_stop_range=(min(50e3, H_max), H_max))
    _logger.debug("Descending loop contains %s points", len(sol.HMB_major_descending))
    vis.plot(sol, f"{coef}{PLOT_FILE_SUFFIX}", bh_curve)

    H_c, B_r, BH_max = bh.extract_H_c_B_r_BH_max_from_major_descending_loop(
        np.delete(sol.HMB_major_descending[::-1], 1, axis=1)
    )
    _logger.info("Predicted parameters: H_c=%.6f A/m, B_r=%.6f T, BH_max=%.3f J/m^3", H_c, B_r, BH_max)


def main() -> None:
    try:
        _setup_logging()
        _cleanup()
        np.seterr(divide="raise", over="raise")
        unnamed, named = _parse_args(sys.argv[1:])
        run(*unnamed, **named)  # type: ignore
    except KeyboardInterrupt:
        _logger.info("Interrupted")
        exit(1)
    except Exception as ex:
        _logger.error("Failure: %s: %s", type(ex).__name__, ex)
        _logger.debug("Failure: %s", ex, exc_info=True)
        exit(1)
    finally:
        # Wait for the background tasks (like plotting) to finish.
        BG_EXECUTOR.shutdown()


def _perhaps(x: T | None, default: T) -> T:
    return default if x is None else x


def _parse_args(args: Iterable[str]) -> tuple[
    list[str],
    dict[str, int | float | str],
]:
    unnamed: list[str] = []
    named: dict[str, int | float | str] = {}
    for arg in args:
        if "=" in arg:
            key, value = arg.split("=", 1)
            try:
                named[key] = int(value, 0)
            except ValueError:
                try:
                    named[key] = float(value)
                except ValueError:
                    named[key] = value
        else:
            unnamed.append(arg)
    return unnamed, named


def _setup_logging() -> None:
    logging.getLogger().setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-3.3s %(name)s: %(message)s"))
    logging.getLogger().addHandler(console_handler)

    file_handler = logging.FileHandler("jafit.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logging.getLogger("numpy").setLevel(logging.WARNING)
    logging.getLogger("scipy").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def _cleanup() -> None:
    for f in Path.cwd().glob(f"*{PLOT_FILE_SUFFIX}"):
        f.unlink(missing_ok=True)


_logger = logging.getLogger(__name__.replace("__", ""))
