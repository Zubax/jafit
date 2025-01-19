# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

"""
Jiles-Atherton system identification tool: Given a BH curve, finds the Jiles-Atherton model coefficients.
Refer to the README.md for the usage instructions.
"""

from __future__ import annotations

import sys
import time
import dataclasses
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar, Iterable, Any
import logging
from pathlib import Path
import numpy as np
from .ja import Solution, Coef, solve, SolverError
from .mag import HysteresisLoop, extract_H_c_B_r_BH_max, mu_0, hm_to_hj
from .opt import fit_global, fit_local, make_objective_function
from . import loss, io, vis


PLOT_FILE_SUFFIX = ".jafit.png"
CURVE_FILE_SUFFIX = ".jafit.tab"

OUTPUT_SAMPLE_COUNT = 1000


class LimitedBacklogThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, backlog_capacity: int, *args: Any, **kwargs: Any) -> None:
        import queue

        super(LimitedBacklogThreadPoolExecutor, self).__init__(*args, **kwargs)
        # noinspection PyTypeChecker
        self._work_queue = queue.Queue(maxsize=backlog_capacity)  # type: ignore


BG_EXECUTOR = LimitedBacklogThreadPoolExecutor(max_workers=1, backlog_capacity=10)
"""
We only need one worker.
The backlog is limited to manage peak memory utilization (solutions are kept in memory until plotted)
and to avoid losing much data should the process crash.
"""

T = TypeVar("T")


def make_callback(
    file_name_prefix: str,
    ref: HysteresisLoop,
    plot_failed: bool = True,
) -> Callable[
    [int, Coef, tuple[Solution, float] | Exception],
    None,
]:

    def cb(epoch: int, coef: Coef, result: tuple[Solution, float] | Exception) -> None:
        # Keep in mind that this callback itself may be invoked from a different thread.
        def bg() -> None:
            try:
                started_at = time.monotonic()
                plot_file: str | None
                match result:
                    case (sol, loss_value) if isinstance(sol, Solution) and isinstance(loss_value, float):
                        plot_file = f"{file_name_prefix}_#{epoch:05}_{loss_value:.6f}_{coef}{PLOT_FILE_SUFFIX}"
                        plot(sol, ref, coef, plot_file)

                    case SolverError() as ex if ex.partial_curve is not None and plot_failed:
                        plot_file = f"{file_name_prefix}_#{epoch:05}_{type(ex).__name__}_{coef}{PLOT_FILE_SUFFIX}"
                        plot_error(ex, coef, plot_file)

                    case _:
                        plot_file = None

                _logger.debug("Plotting %r took %.0f ms", plot_file, (time.monotonic() - started_at) * 1e3)
            except Exception as ex:
                _logger.error("Failed to plot: %s: %s", type(ex).__name__, ex)
                _logger.debug("Failed to plot: %s", ex, exc_info=True)

        # Plotting can take a while, so we do it in a background thread.
        # The plotting library will release the GIL.
        BG_EXECUTOR.submit(bg)

    return cb


def do_fit(
    ref: HysteresisLoop,
    *,
    c_r: float | None,
    M_s: float | None,
    a: float | None,
    k_p: float | None,
    alpha: float | None,
    H_max: float,
    max_evaluations_per_stage: int,
    skip_stages: int,
    plot_failed: bool,
    fast: bool,
) -> Coef:
    if fast:
        _logger.warning("⚠ Fast mode is enabled; the solver may produce inaccurate results or fail to converge.")

    if len(ref.descending) == 0:
        raise ValueError("The reference descending curve is empty")
    H_c, B_r, BH_max = extract_H_c_B_r_BH_max(ref.descending)
    _logger.info("Given: %s; derived parameters: H_c=%.6f A/m, B_r=%.6f T, BH_max=%.3f J/m^3", ref, H_c, B_r, BH_max)
    M_s_min = B_r / mu_0  # Heuristic: B=mu_0*(H+M); H=0; B=mu_0*M; hence, M_s>=B_r/mu_0
    _logger.info("Derived minimum M_s: %.6f A/m", M_s_min)

    # Interpolate the reference curve such that the sample points are equally spaced to improve the
    # behavior of the nearest-point loss function. This is not needed for the other loss functions.
    ref_original = ref
    ref_interpolated = ref.interpolate_equidistant(300)
    del ref

    # Display the interpolation result for diagnostics and visibility.
    vis.plot(
        [
            ("M(H) interpolated descending", ref_interpolated.descending, vis.Style.scatter, vis.Color.black),
            ("M(H) interpolated ascending", ref_interpolated.ascending, vis.Style.scatter, vis.Color.blue),
            ("M(H) original descending", ref_original.descending, vis.Style.scatter, vis.Color.red),
            ("M(H) original ascending", ref_original.ascending, vis.Style.scatter, vis.Color.magenta),
        ],
        "Reference curve interpolation",
        f"reference_interpolation{PLOT_FILE_SUFFIX}",
        axes_labels=("H [A/m]", "B [T]"),
        square_aspect_ratio=True,  # Same aspect ratio is required to check that the points are equidistant.
    )

    # Ensure the interpolation did not cause nontrivial distortion.
    interp_H_c, interp_B_r, interp_BH_max = extract_H_c_B_r_BH_max(ref_interpolated.descending)
    _logger.debug(
        "After interpolation: H_c=%.6f A/m, B_r=%.6f T, BH_max=%.3f J/m^3", interp_H_c, interp_B_r, interp_BH_max
    )
    assert np.isclose(interp_H_c, H_c, rtol=0.1, atol=1e-3), f"{interp_H_c} != {H_c}"
    assert np.isclose(interp_B_r, B_r, rtol=0.1, atol=1e-3), f"{interp_B_r} != {B_r}"
    assert np.isclose(interp_BH_max, BH_max, rtol=0.1, atol=1e-3), f"{interp_BH_max} != {BH_max}"

    # Initialize the coefficients and their bounds.
    coef = Coef(
        c_r=_perhaps(c_r, 0.1),
        M_s=_perhaps(M_s, M_s_min * 1.001),  # Optimizers tend to be unstable if parameters are too close to the bounds
        a=_perhaps(a, 1e3),
        k_p=_perhaps(k_p, max(H_c, 1.0)),  # For soft materials, k_p≈H_ci. For others this is still a good guess.
        alpha=_perhaps(alpha, 0.001),
    )
    x_min = Coef(c_r=1e-9, M_s=M_s_min, a=1e-6, k_p=1e-6, alpha=1e-9)
    # TODO: better way of setting the upper bounds?
    x_max = Coef(c_r=0.999999, M_s=3e6, a=3e6, k_p=3e6, alpha=0.999999)
    _logger.info("Initial, minimum, and maximum coefficients:\n%s\n%s\n%s", coef, x_min, x_max)

    # Ensure the swept H-range is large enough.
    # This is to ensure that the saturation detection heuristic does not mistakenly terminate the sweep too early.
    H_stop_range = float(
        max(
            np.max(np.abs(ref_original.H_range)),
            H_c * 2,
            M_s_min * 0.1,
        )
    ), float(H_max)
    _logger.info("H amplitude range: %s [A/m]", H_stop_range)

    if (H_c > 100 and B_r > 0.1) and skip_stages < 1:
        _logger.info("Demag knee detected; performing initial H_c|B_r|BH_max optimization")
        coef = fit_global(
            x_0=coef,
            x_min=x_min,
            x_max=x_max,
            obj_fn=make_objective_function(
                ref_original,  # Here we're using the non-interpolated curve.
                loss.demag_key_points,
                H_stop_range=H_stop_range,
                stop_loss=0.01,  # Fine adjustment is meaningless the loss fun is crude here.
                stop_evals=max_evaluations_per_stage,
                callback=make_callback("0_initial", ref_original, plot_failed=plot_failed),
                solver_extra_args=dict(fast=fast),
            ),
            tolerance=1e-3,
        )
        _logger.info(f"Intermediate result:\n%s", coef)
    else:
        _logger.info("Skipping initial optimization")

    if skip_stages < 2:
        coef = fit_global(
            x_0=coef,
            x_min=x_min,
            x_max=x_max,
            obj_fn=make_objective_function(
                ref_interpolated,
                loss.nearest,
                H_stop_range=H_stop_range,
                stop_evals=max_evaluations_per_stage,
                callback=make_callback("1_global", ref_interpolated, plot_failed=plot_failed),
                solver_extra_args=dict(fast=fast),
            ),
            tolerance=1e-7,
        )
        _logger.info(f"Intermediate result:\n%s", coef)

    coef = fit_local(
        x_0=coef,
        x_min=x_min,
        x_max=x_max,
        obj_fn=make_objective_function(
            ref_interpolated,
            loss.nearest,
            H_stop_range=H_stop_range,
            stop_evals=max_evaluations_per_stage,
            callback=make_callback("2_local", ref_interpolated, plot_failed=plot_failed),
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


def plot(
    sol: Solution,
    ref: HysteresisLoop | None,
    coef: Coef,
    plot_file: Path | str,
    subtitle: str | None = None,
) -> None:
    S, C = vis.Style, vis.Color
    specs = [
        ("J(H) JA virgin", hm_to_hj(sol.virgin), S.line, C.gray),
        ("J(H) JA descending", hm_to_hj(sol.descending), S.line, C.black),
        ("J(H) JA ascending", hm_to_hj(sol.ascending), S.line, C.blue),
    ]
    if ref:
        specs.append(("J(H) reference descending", hm_to_hj(ref.descending), S.scatter, C.red))
        specs.append(("J(H) reference ascending", hm_to_hj(ref.ascending), S.scatter, C.magenta))
    title = str(coef)
    if subtitle:
        title += f"\n{subtitle}"
    vis.plot(specs, title, plot_file, axes_labels=("H [A/m]", "B [T]"))


def plot_error(ex: SolverError, coef: Coef, plot_file: Path | str) -> None:
    if ex.partial_curve is None:
        _logger.debug("No partial curve to plot: %s", ex)
        return
    specs = [
        ("M(H)", ex.partial_curve, vis.Style.line, vis.Color.black),
    ]
    title = f"{coef}\n{type(ex).__name__}\n{ex}"
    vis.plot(specs, title, plot_file, axes_labels=("H [A/m]", "M [A/m]"))
    _logger.debug("%s max(|dH|)=%s", type(ex).__name__, np.abs(np.diff(ex.partial_curve[:, 0])).max())


def run(
    ref_file_path: str | None = None,
    *unnamed_args: str,
    c_r: float | None = None,
    M_s: float | None = None,
    a: float | None = None,
    k_p: float | None = None,
    alpha: float | None = None,
    H_min: float = 0,
    H_max: float = 5e6,
    effort: int = 10**7,
    skip_stages: int = 0,
    plot_failed: bool = False,
    fast: bool = False,
    **named_args: dict[str, int | float | str],
) -> None:
    if unnamed_args or named_args:
        raise ValueError(f"Unexpected arguments:\nunnamed: {unnamed_args}\nnamed: {named_args}")

    ja_dict = {
        k: (float(v) if v is not None else None) for k, v in dict(c_r=c_r, M_s=M_s, a=a, k_p=k_p, alpha=alpha).items()
    }
    H_max = float(H_max)
    effort = int(effort)
    effort = int(effort)
    skip_stages = int(skip_stages)
    plot_failed = bool(plot_failed)
    fast = bool(fast)

    ref = io.load(Path(ref_file_path)) if ref_file_path else None
    if ref is not None:
        _logger.info("Fitting %s with starting parameters: %s", ref, ja_dict)
        coef = do_fit(
            ref,
            **ja_dict,
            H_max=H_max,
            max_evaluations_per_stage=effort,
            skip_stages=skip_stages,
            plot_failed=plot_failed,
            fast=fast,
        )
        # noinspection PyTypeChecker
        print(*(f"{k}={v}" for k, v in dataclasses.asdict(coef).items()))
    else:
        _logger.info("No BH curve given, fitting will not be performed")
        if any(x is None for x in ja_dict.values()):
            raise ValueError(f"Supplied coefficients are incomplete, and optimization is not requested: {ja_dict}")
        coef = Coef(**ja_dict)  # type: ignore

    # Solve with the coefficients and plot the results.
    _logger.info("Solving and plotting: %s", coef)
    if H_min < 1e-9:
        H_min = max(coef.k_p * 2, 100e3)  # For soft materials, k_p≈H_ci; this is a good guess for H_min.
    H_stop_range = H_min, H_max
    try:
        sol = solve(coef, H_stop_range=H_stop_range)
    except SolverError as ex:
        plot_error(ex, coef, f"{type(ex).__name__}.{coef}{PLOT_FILE_SUFFIX}")
        raise
    loop = HysteresisLoop(descending=sol.descending, ascending=sol.ascending)
    _logger.debug("Solved loop: %s", loop)

    # Extract the key parameters from the descending loop.
    H_c, B_r, BH_max = extract_H_c_B_r_BH_max(sol.descending)
    _logger.info("Predicted parameters: H_c=%.6f A/m, B_r=%.6f T, BH_max=%.3f J/m^3", H_c, B_r, BH_max)

    # noinspection PyTypeChecker
    plot(sol, ref, coef, f"{coef}{PLOT_FILE_SUFFIX}", subtitle=f"H_c={H_c:.0f} B_r={B_r:.3f} BH_max={BH_max:.0f}")

    # Save the BH curves.
    decimated_loop = loop.decimate(OUTPUT_SAMPLE_COUNT)
    io.save(Path(f"B(H).loop{CURVE_FILE_SUFFIX}"), decimated_loop)
    io.save(Path(f"B(H).desc{CURVE_FILE_SUFFIX}"), decimated_loop.descending)
    io.save(Path(f"B(H).virgin{CURVE_FILE_SUFFIX}"), sol.virgin[:: max(1, len(sol.virgin) // OUTPUT_SAMPLE_COUNT)])


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
    console_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-3.3s %(name)s: %(message)s", "%H:%M:%S"))
    logging.getLogger().addHandler(console_handler)

    file_handler = logging.FileHandler("jafit.log", mode="w", encoding="utf-8")
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logging.getLogger("numpy").setLevel(logging.WARNING)
    logging.getLogger("scipy").setLevel(logging.WARNING)
    logging.getLogger("numba").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.ERROR)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def _cleanup() -> None:
    for f in Path.cwd().glob(f"*{PLOT_FILE_SUFFIX}"):
        f.unlink(missing_ok=True)
    for f in Path.cwd().glob(f"*{CURVE_FILE_SUFFIX}"):
        f.unlink(missing_ok=True)


_logger = logging.getLogger(__name__.replace("__", ""))
