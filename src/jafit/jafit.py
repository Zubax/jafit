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
from typing import Callable, TypeVar, Iterable, Any, overload
import logging
from pathlib import Path
import numpy as np
from .ja import Solution, Coef, solve, SolverError, Model
from .mag import HysteresisLoop, extract_H_c_B_r_BH_max, mu_0, hm_to_hj
from .opt import fit_global, fit_local, make_objective_function
from . import loss, io, vis, __version__


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
    H_amp_max: float | None,
    interpolate_points: int | None,
    max_evaluations_per_stage: int | None,
    stage: int,
    plot_failed: bool,
    fast: bool,
    quiet: bool,
) -> tuple[
    Coef,
    tuple[float, float] | float,
]:
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
    coef = Coef(
        c_r=_perhaps(c_r, 0.1),
        M_s=_perhaps(M_s, M_s_min * 1.001),  # Optimizers tend to be unstable if parameters are too close to the bounds
        a=_perhaps(a, 1e3),
        k_p=_perhaps(k_p, max(H_c, 1.0)),  # For soft materials, k_p≈H_ci. For others this is still a good guess.
        alpha=_perhaps(alpha, 0.001),
    )
    x_min = Coef(c_r=1e-12, M_s=M_s_min, a=1e-6, k_p=1e-6, alpha=1e-12)
    # TODO: We need a better way of setting the upper bound. The sensible limits also depend on the model used.
    x_max = Coef(c_r=0.999999999, M_s=3e6, a=3e6, k_p=3e6, alpha=10.0)
    _logger.info("Initial, minimum, and maximum coefficients:\n%s\n%s\n%s", coef, x_min, x_max)

    # Ensure that the saturation detection heuristic does not mistakenly terminate the sweep too early.
    # If we have a full hysteresis loop, simply limit the H-range to that; this speeds up optimization considerably
    # because we can quickly weed out materials that don't behave as expected in the specified loop. The provided
    # loop in this case doesn't need to be the major one, too!
    if not H_amp_max:
        _logger.warning("H_amp_max is not specified; using a heuristic")
    H_stop: tuple[float, float] | float
    if ref.is_full:
        # If we have the full loop, simply replicate its H-range. The loop may be a minor one.
        H_stop = max(float(np.abs(ref.H_range).max()), H_amp_max or 0)
    else:
        # If we only have a part of the loop, assume that part belongs to the major loop.
        # We have to employ heuristics to determine when to stop the sweep.
        H_stop = float(
            max(
                np.abs(ref.H_range).max(),
                H_c * 2,
                M_s_min * 0.1,
            )
        ), float(H_amp_max or max(100e3, M_s_min * 2, H_c * 4))
    _logger.info("H amplitude range: %s [A/m]", H_stop)

    # Run the optimizer.
    if (H_c > 100 and B_r > 0.1) and stage < 1:
        _logger.info("Demag knee detected; performing initial H_c|B_r|BH_max optimization")
        coef = fit_global(
            x_0=coef,
            x_min=x_min,
            x_max=x_max,
            obj_fn=make_objective_function(
                ref,  # Here we're using the non-interpolated curve.
                lambda c: solve(model, c, H_stop, fast=True),  # The initial exploration always uses the fast mode.
                loss.demag_key_points,
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
                ref_interpolated,
                lambda c: solve(model, c, H_stop, fast=fast),
                loss.nearest,
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
            ref_interpolated,
            lambda c: solve(model, c, H_stop),  # Fine-tuning cannot use fast mode.
            loss.nearest,
            stop_evals=max_evaluations_per_stage or 10**5,
            callback=make_callback("2_local", ref_interpolated, plot_failed=plot_failed),
            quiet=quiet,
        ),
    )

    # Emit a warning if the final coefficients are close to the bounds.
    rtol, atol = 1e-6, 1e-9
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
    H_amp_min: float | None,
    H_amp_max: float | None,
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
            H_amp_max=H_amp_max,
            interpolate_points=interpolate_points,
            max_evaluations_per_stage=effort,
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
        H_amp_min = H_amp_min or max(coef.k_p * 2, 100e3)  # In soft materials k_p≈H_ci
        H_amp_max = H_amp_max or max(coef.M_s * 2, 1e6)
        H_stop = H_amp_min, H_amp_max

    # Solve with the coefficients and plot the results.
    _logger.info("Solving and plotting using %s model: %s; H amplitude: %s", model.name.lower(), coef, H_stop)
    try:
        sol = solve(model, coef, H_stop=H_stop)
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
        if model is None:
            raise ValueError(
                f"Model name not understood: {model_name!r}. Choose one: {', '.join(x.name.lower() for x in Model)}"
            )

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

        assert isinstance(model, Model)
        run(
            model=model,
            ref=ref,
            cf=coef,
            H_amp_min=_param(named, "H_amp_min", float),
            H_amp_max=_param(named, "H_amp_max", float),
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
