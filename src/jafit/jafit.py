# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

"""
Jiles-Atherton system identification tool: Given a BH curve, finds the Jiles-Atherton model coefficients.
Refer to the README.md for the usage instructions.
"""

from __future__ import annotations

import sys
import time
import dataclasses
from itertools import cycle
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, TypeVar, Iterable, Any, overload
import logging
from pathlib import Path
import numpy as np
from .ja import Solution, Coef, solve, SolverError, Model
from .mag import HysteresisLoop, extract_H_c_B_r_BH_max, hm_to_hj
from .opt import fit_global, fit_local, make_objective_function
from . import loss, io, vis, interactive, __version__


PLOT_FILE_SUFFIX = ".jafit.png"
CURVE_FILE_SUFFIX = ".jafit.tab"

OUTPUT_SAMPLE_COUNT = 10000


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

                    case SolverError() as ex if ex.partial_curves and plot_failed:
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
    model: Model,
    c_r: float | None,
    M_s: float | None,
    a: float | None,
    k_p: float | None,
    alpha: float | None,
    M_s_min: float | None,
    M_s_max: float | None,
    H_amp_min: float | None,
    H_amp_max: float | None,
    interpolate_points: int | None,
    max_evaluations_per_stage: int | None,
    priority_region_error_gain: float | None,
    stage: int,
    plot_failed: bool,
    fast: bool,
    quiet: bool,
) -> tuple[Coef, tuple[float, float]]:
    if fast:
        _logger.warning("⚠ Fast mode is enabled; the solver may produce inaccurate results or fail to converge.")

    if len(ref.descending) == 0:
        raise ValueError("The reference descending curve is empty")
    H_c, B_r, BH_max = extract_H_c_B_r_BH_max(ref.descending)
    _logger.info("Given: %s; derived parameters: H_c=%.6f A/m, B_r=%.6f T, BH_max=%.3f J/m^3", ref, H_c, B_r, BH_max)

    # Interpolate the reference curve such that the sample points are equally spaced to improve the
    # behavior of the nearest-point loss function. This is not needed for the other loss functions.
    if interpolate_points:
        if interpolate_points < 10:
            raise ValueError(f"Interpolation point count is too low: {interpolate_points}")
        ref_interpolated = ref.interpolate_equidistant(interpolate_points)
        # Display the interpolation result for diagnostics and visibility.
        interpolation_plot_specs = [
            ("M(H) interpolated descending", ref_interpolated.descending, vis.Style.scatter, vis.Color.black),
            ("M(H) interpolated ascending", ref_interpolated.ascending, vis.Style.scatter, vis.Color.blue),
            ("M(H) original descending", ref.descending, vis.Style.scatter, vis.Color.red),
            ("M(H) original ascending", ref.ascending, vis.Style.scatter, vis.Color.magenta),
        ]
        vis.plot(
            interpolation_plot_specs,
            "Reference curve interpolation with 1:1 aspect ratio",
            f"reference_interpolation_square{PLOT_FILE_SUFFIX}",
            axes_labels=("H [A/m]", "B [T]"),
            square_aspect_ratio=True,  # Same aspect ratio is required to check that the points are equidistant.
        )
        vis.plot(
            interpolation_plot_specs,
            "Reference curve interpolation",
            f"reference_interpolation{PLOT_FILE_SUFFIX}",
            axes_labels=("H [A/m]", "B [T]"),
        )
        # Ensure the interpolation did not cause nontrivial distortion.
        interp_H_c, interp_B_r, interp_BH_max = extract_H_c_B_r_BH_max(ref_interpolated.descending)
        _logger.debug(
            "After interpolation: H_c=%.6f A/m, B_r=%.6f T, BH_max=%.3f J/m^3", interp_H_c, interp_B_r, interp_BH_max
        )
        assert np.isclose(interp_H_c, H_c, rtol=0.1, atol=1e-3), f"{interp_H_c} != {H_c}"
        assert np.isclose(interp_B_r, B_r, rtol=0.1, atol=1e-3), f"{interp_B_r} != {B_r}"
        assert np.isclose(interp_BH_max, BH_max, rtol=0.1, atol=1e-3), f"{interp_BH_max} != {BH_max}"
    else:
        ref_interpolated = ref

    # Initialize the coefficients and their bounds.
    if M_s_min is None:
        M_s_min = float(  # Saturation magnetization cannot be less than the values seen in the reference curve.
            max(
                np.abs(ref.descending[:, 1]).max(initial=0),
                np.abs(ref.ascending[:, 1]).max(initial=0),
            )
        )
    assert M_s_min is not None
    M_s_max = float(  # Maximum cannot be less than the minimum. If they are equal, assume M_s is known precisely.
        max(
            M_s_min,
            M_s_max if M_s_max is not None else max(M_s_min * 1.6, 2e6),
        )
    )
    _logger.info("Using M_s_min=%f M_s_max=%f", M_s_min, M_s_max)
    assert M_s_max is not None
    coef = Coef(
        c_r=_perhaps(c_r, 0.1),
        M_s=_perhaps(M_s, (M_s_min + M_s_max) * 0.5),
        a=_perhaps(a, 10e3),
        k_p=_perhaps(k_p, max(H_c, 1.0)),  # For soft materials, k_p≈H_ci. For others this is still a good guess.
        alpha=_perhaps(alpha, 0.0001),
    )
    x_min = Coef(c_r=1e-12, M_s=M_s_min, a=1e-6, k_p=1e-6, alpha=1e-12)
    x_max = Coef(c_r=0.999999999, M_s=M_s_max, a=3e6, k_p=3e6, alpha=10.0)
    _logger.info("Initial, minimum, and maximum coefficients:\n%s\n%s\n%s", coef, x_min, x_max)

    # Determine the H amplitude range.
    if H_amp_min is None:
        # By default, simply use the peak value from the reference dataset.
        # This enables simple treatment of minor loops, where we just set H_amp_max=0, and then
        # H_amp_min=H_amp_max=max(abs(H_ref)), thus we repeat the excitation from the reference dataset.
        # Using anything more clever here would make this simple case more complex and we don't want that.
        H_amp_min = float(np.abs(ref.H_range).max())  # Like in the reference dataset.
    if H_amp_max is None:
        # We need to ensure that we can push the material into saturation while not wasting too much time
        # trying to solve nonsaturable materials with very low permeability.
        H_amp_max = max(100e3, M_s_max * 2, H_c * 5)  # Being clever.
    H_amp_max = max(H_amp_max, H_amp_min)
    H_stop = H_amp_min, H_amp_max
    _logger.info("H amplitude range: %s [A/m]", H_stop)
    assert H_stop[0] <= H_stop[1]

    # Loss function parameters.
    priority_region_error_gain = priority_region_error_gain or 1.0

    # Run the optimizer.
    if (H_c > 100 and B_r > 0.1) and stage < 1:
        _logger.info("Demag knee detected; performing initial H_c|B_r|BH_max optimization")
        coef = fit_global(
            x_0=coef,
            x_min=x_min,
            x_max=x_max,
            obj_fn=make_objective_function(
                lambda c: solve(model, c, H_stop, fast=True),  # The initial exploration always uses the fast mode.
                loss.make_demag_key_points(ref),  # Here we're using the non-interpolated curve.
                stop_loss=0.01,  # Fine adjustment is meaningless the loss fun is crude here.
                stop_evals=max_evaluations_per_stage or 10**5,
                callback=make_callback("0_initial", ref, plot_failed=plot_failed),
                quiet=quiet,
            ),
            tolerance=1e-3,
        )
        _logger.info(f"Intermediate result:\n%s", coef)
    else:
        _logger.info("Skipping initial optimization")

    if stage < 2:
        coef = fit_global(
            x_0=coef,
            x_min=x_min,
            x_max=x_max,
            obj_fn=make_objective_function(
                lambda c: solve(model, c, H_stop, fast=fast),
                loss.make_nearest(ref_interpolated, priority_region_error_gain=priority_region_error_gain),
                stop_evals=max_evaluations_per_stage or 10**7,
                callback=make_callback("1_global", ref_interpolated, plot_failed=plot_failed),
                quiet=quiet,
            ),
            tolerance=1e-7,
        )
        _logger.info(f"Intermediate result:\n%s", coef)

    coef = fit_local(
        x_0=coef,
        x_min=x_min,
        x_max=x_max,
        obj_fn=make_objective_function(
            lambda c: solve(model, c, H_stop),  # Fine-tuning cannot use fast mode.
            loss.make_nearest(ref_interpolated, priority_region_error_gain=priority_region_error_gain),
            stop_evals=max_evaluations_per_stage or 10**5,
            callback=make_callback("2_local", ref_interpolated, plot_failed=plot_failed),
            quiet=quiet,
        ),
    )

    # Emit a warning if the final coefficients are close to the bounds.
    rtol, atol = 0.01, 1e-6
    # noinspection PyTypeChecker
    for k, v in dataclasses.asdict(coef).items():
        lo, hi = x_min.__getattribute__(k), x_max.__getattribute__(k)
        if np.isclose(v, lo, rtol, atol) or np.isclose(v, hi, rtol, atol):
            _logger.warning("Final %s=%.9f is close to the bounds [%.9f, %.9f]", k, v, lo, hi)

    return coef, H_stop


def plot(
    sol: Solution,
    ref: HysteresisLoop | None,
    coef: Coef,
    plot_file: Path | str,
    subtitle: str | None = None,
) -> None:
    S, C = vis.Style, vis.Color
    color = iter(cycle([C.gray, C.black, C.black, C.red, C.blue]))
    specs = [
        (
            f"J(H) JA branch #{idx}",
            hm_to_hj(curve),
            S.line,
            next(color),
        )
        for idx, curve in enumerate(sol.branches)
    ]
    if ref:
        specs.append(("J(H) reference descending", hm_to_hj(ref.descending), S.scatter, C.orange))
        specs.append(("J(H) reference ascending", hm_to_hj(ref.ascending), S.scatter, C.magenta))
    title = str(coef)
    if subtitle:
        title += f"\n{subtitle}"
    vis.plot(specs, title, plot_file, axes_labels=("H [A/m]", "B [T]"))


def plot_error(ex: SolverError, coef: Coef, plot_file: Path | str) -> None:
    if not ex.partial_curves:
        _logger.debug("No partial curve to plot: %s", ex)
        return
    colors = [e for e in vis.Color]
    specs = [
        (
            f"M(H) #{idx}",
            curve,
            vis.Style.line,
            colors[idx % len(colors)],
        )
        for idx, curve in enumerate(ex.partial_curves)
    ]
    title = f"{coef}\n{type(ex).__name__}\n{ex}"
    vis.plot(specs, title, plot_file, axes_labels=("H [A/m]", "M [A/m]"))


def run(
    model: Model,
    ref: HysteresisLoop | None,
    cf: dict[str, float | None],
    M_s_min: float | None,
    M_s_max: float | None,
    H_amp_min: float | None,
    H_amp_max: float | None,
    priority_region_error_gain: float | None,
    interpolate_points: int | None,
    effort: int | None,
    stage: int,
    plot_failed: bool,
    fast: bool,
    quiet: bool,
) -> None:
    H_stop: float | tuple[float, float]
    if ref is not None:
        _logger.info(
            "Fitting %s using %s model with starting parameters: %s; effort=%s", ref, model.name.lower(), cf, effort
        )
        coef, H_stop = do_fit(
            ref,
            model=model,
            **cf,
            M_s_min=M_s_min,
            M_s_max=M_s_max,
            H_amp_min=H_amp_min,
            H_amp_max=H_amp_max,
            interpolate_points=interpolate_points,
            max_evaluations_per_stage=effort,
            priority_region_error_gain=priority_region_error_gain,
            stage=stage,
            plot_failed=plot_failed,
            fast=fast,
            quiet=quiet,
        )
        # noinspection PyTypeChecker
        print(*(f"{k}={v}" for k, v in dataclasses.asdict(coef).items()))
    else:
        if any(x is None for x in cf.values()):
            raise ValueError(f"Supplied coefficients are incomplete, and optimization is not requested: {cf}")
        coef = Coef(**cf)  # type: ignore
        H_amp_min = H_amp_min or max(coef.k_p * 2, 10e3)  # In soft materials k_p≈H_ci
        H_amp_max = H_amp_max or max(coef.M_s * 2, 1e6)
        H_stop = H_amp_min, H_amp_max

    # Solve with the coefficients and plot the results.
    _logger.info("Solving and plotting using %s model: %s; H amplitude: %s", model.name.lower(), coef, H_stop)
    try:
        sol = solve(model, coef, H_stop=H_stop)
    except SolverError as ex:
        plot_error(ex, coef, f"{type(ex).__name__}.{coef}{PLOT_FILE_SUFFIX}")
        raise
    loop = HysteresisLoop(descending=sol.last_descending[::-1], ascending=sol.last_ascending)
    _logger.debug("Solved loop: %s", loop)

    # Extract the key parameters from the descending loop.
    H_c, B_r, BH_max = extract_H_c_B_r_BH_max(loop.descending)
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
        _logger.debug("jafit v%s invoked as:\n%s", __version__, " ".join(sys.argv))
        _cleanup()
        np.seterr(divide="raise", over="raise")
        unnamed, named = _parse_args(sys.argv[1:])
        if unnamed:
            raise ValueError(f"Unexpected unnamed arguments: {unnamed}")

        # Parse the model name.
        model: Model | None = None
        model_name = _param(named, "model", str, "").strip().upper()
        if model_name:
            for enum_item in Model:
                if enum_item.name.upper().startswith(model_name):
                    model = enum_item
                    break

        # Load the reference curve.
        ref: HysteresisLoop | None = None
        if ref_file_str := _param(named, "ref", str, ""):
            ref = io.load(Path(ref_file_str))

        # Load the coefficients.
        coef = dict(
            c_r=_param(named, "c_r", float),
            M_s=_param(named, "M_s", float),
            a=_param(named, "a", float),
            k_p=_param(named, "k_p", float),
            alpha=_param(named, "alpha", float),
        )

        M_s_min = _param(named, "M_s_min", float)
        M_s_max = _param(named, "M_s_max", float)
        H_amp_min = _param(named, "H_amp_min", float)
        H_amp_max = _param(named, "H_amp_max", float)
        if _param(named, "interactive", bool, False):
            model = model or Model.VENKATARAMAN
            initial_coef = Coef(
                c_r=coef["c_r"] or 0.1,
                M_s=coef["M_s"] or M_s_min or M_s_max or 1e6,
                a=coef["a"] or 100e3,
                k_p=coef["k_p"] or 100e3,
                alpha=coef["alpha"] or 0.1,
            )
            if H_amp_min is None:
                H_amp_min = max(np.abs(ref.H_range)) if ref is not None else initial_coef.k_p * 2
            if H_amp_max is None:
                H_amp_max = max(1e6, initial_coef.M_s * 2, initial_coef.k_p * 5)  # k_p≈H_ci for soft materials
            interactive.run(ref, model, initial_coef, (H_amp_min, H_amp_max))
            return

        if model is None:
            raise ValueError(
                f"Model name not understood: {model_name!r}. Choose one: {', '.join(x.name.lower() for x in Model)}"
            )
        assert isinstance(model, Model)
        run(
            model=model,
            ref=ref,
            cf=coef,
            M_s_min=M_s_min,
            M_s_max=M_s_max,
            H_amp_min=H_amp_min,
            H_amp_max=H_amp_max,
            priority_region_error_gain=_param(named, "preg", float),
            interpolate_points=_param(named, "interpolate", int),
            effort=_param(named, "effort", int),
            stage=_param(named, "stage", int, 0),
            plot_failed=_param(named, "plot_failed", bool, False),
            fast=_param(named, "fast", bool, False),
            quiet=_param(named, "quiet", bool, False, last=True),
        )
    except KeyboardInterrupt:
        _logger.info("Interrupted")
        _logger.debug("Interruption stack trace", exc_info=True)
        exit(1)
    except Exception as ex:
        _logger.error("Failure: %s: %s", type(ex).__name__, ex)
        _logger.debug("Failure: %s", ex, exc_info=True)
        exit(1)
    finally:
        # Wait for the background tasks (like plotting) to finish.
        BG_EXECUTOR.shutdown()


ParamType = Callable[[int | float | str], T]
_sentinel = object()


@overload
def _param(d: dict[str, int | float | str], name: str, ty: ParamType[T], *, last: bool = False) -> T | None: ...
@overload
def _param(d: dict[str, int | float | str], name: str, ty: ParamType[T], default: T, last: bool = False) -> T: ...
def _param(
    d: dict[str, int | float | str],
    name: str,
    ty: ParamType[T],
    default: Any = _sentinel,
    last: bool = False,
) -> T | None:
    try:
        v = d.pop(name)
    except KeyError:
        if default is _sentinel:
            return None
        return default  # type: ignore
    finally:
        if last and d:
            raise ValueError(f"Unexpected named arguments: {d}")
    return ty(v)


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
