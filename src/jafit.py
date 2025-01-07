#!/usr/bin/env python3
# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

"""
This is a simple utility that fits the Jiles-Atherton (JA) model coefficients for a given BH curve.
The JA model used here follows that of COMSOL Multiphysics; see the enclosed PDF with the relevant excerpt
from the COMSOL user reference.

The following coefficients are defined; per the model definition, all of them can be scalars or 3x3 matrices:

Symbol  Description                                                         Range           Unit
c_r     magnetization reversibility (1 for purely anhysteretic material)    (0, 1]          dimensionless
M_s     saturation magnetization                                            positive real   ampere/meter
a       domain wall density                                                 positive real   ampere/meter
k_p     pinning loss                                                        positive real   ampere/meter
alpha   interdomain coupling                                                non-negative    dimensionless
"""

from __future__ import annotations
import os
import sys
import copy
from logging import getLogger, basicConfig
from pathlib import Path
import numpy as np
import numpy.typing as npt
import matplotlib
import ja

NO_DISPLAY = os.name == "posix" and "DISPLAY" not in os.environ
if NO_DISPLAY:  # https://stackoverflow.com/a/45756291/1007777
    matplotlib.use("Agg")


# noinspection PyPep8Naming
def bh_extrapolate(bh_curve: npt.NDArray[np.float64], H: float) -> npt.NDArray[np.float64]:
    """
    Takes BH curve data points and extrapolates its last segment to the specified H value.
    This is in line with how BH curves are commonly treated in magnetic simulation software.
    Returns the extended BH curve (take the edge element if only the extrapolated point is needed).
    """
    if not np.all(np.diff(bh_curve[:, 0]) > 0):
        raise ValueError("Bad BH curve: H is not monotonically increasing")
    if bh_curve[0, -1] >= H:
        raise ValueError("Extrapolation not necessary: the last point of the BH curve is already at or beyond H")
    dB_dH_ref = (bh_curve[-1, 1] - bh_curve[-2, 1]) / (bh_curve[-1, 0] - bh_curve[-2, 0])
    B_ref_ext = bh_curve[-1, 1] + dB_dH_ref * (H - bh_curve[-1, 0])
    return np.vstack([bh_curve, [H, B_ref_ext]])


def visualize(sol: ja.Solution, bh_curve_ref: npt.NDArray[np.float64], max_points: float = 1e4) -> None:
    import matplotlib.pyplot as plt

    fig, (ax_m, ax_b) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    # Plot the curves predicted by the JA model
    for i, fragment in enumerate(sol.H_M_B_segments, start=1):
        n_points = fragment.shape[0]
        if n_points > max_points:  # Select `max_points` evenly spaced indices from the fragment
            indices = np.round(np.linspace(0, n_points - 1, int(max_points))).astype(int)
            fragment = fragment[indices, :]
        H_vals = fragment[:, 0]
        M_vals = fragment[:, 1]
        B_vals = fragment[:, 2]
        ax_m.plot(H_vals, M_vals, label=f"JA fragment {i}")
        ax_b.plot(H_vals, B_vals, label=f"JA fragment {i}")

    # Plot the reference BH curve
    # First, extrapolate the rightmost segment to the max H value of the JA prediction
    H_max = np.max([np.max(frag[:, 0]) for frag in sol.H_M_B_segments])
    bh_curve_ref_ext = bh_extrapolate(bh_curve_ref, H_max)
    ax_b.scatter(bh_curve_ref_ext[:, 0], bh_curve_ref_ext[:, 1], marker="x", label="Reference BH curve data")

    # Configure Magnetization subplot
    ax_m.set_title("Magnetization vs. Field")
    ax_m.set_xlabel("H (ampere/meter)")
    ax_m.set_ylabel("M (ampere/meter)")
    ax_m.legend()
    ax_m.grid(True)

    # Configure Flux Density subplot
    ax_b.set_title("Flux Density vs. Field")
    ax_b.set_xlabel("H (ampere/meter)")
    ax_b.set_ylabel("B (tesla)")
    ax_b.legend()
    ax_b.grid(True)

    # Show the plot
    plt.tight_layout()
    if not NO_DISPLAY:
        plt.show()
    else:
        output_file = "jafit.png"
        _logger.info(f"Saving the plot to {output_file!r}")
        plt.savefig(output_file)


def load_bh_curve(file_path: Path) -> npt.NDArray[np.float64]:
    """
    Returns a matrix of shape (n, 2) where n is the number of data points; the columns are the applied field H
    and the flux density B, respectively.
    """
    bh_file_lines = file_path.read_text().splitlines()
    try:
        [float(x) for x in bh_file_lines[0].split()]
    except ValueError:
        _logger.info("Skipping the first line of the input file, assuming it is the header: %r", bh_file_lines[0])
        bh_file_lines = bh_file_lines[1:]
    bh_data = np.array([[float(x) for x in line.split()] for line in bh_file_lines], dtype=np.float64)
    if bh_data.shape[1] != 2 or bh_data.shape[0] < 2:
        raise ValueError(f"Invalid BH curve data shape: {bh_data.shape}")
    return bh_data


# noinspection PyPep8Naming
def main() -> None:
    basicConfig(
        level="INFO",
        format="%(asctime)s %(levelname)-3.3s %(name)s: %(message)s",
    )

    # Load and validate the BH curve
    bh_curve = load_bh_curve(Path(sys.argv[1]))
    _logger.info("BH curve loaded:\r%s", bh_curve)
    if not np.all(np.diff(bh_curve[:, 0]) > 0):
        raise ValueError("Bad BH curve: H is not monotonically increasing")
    if not np.all(np.diff(bh_curve[:, 1]) > 0):
        raise ValueError("Bad BH curve: B is not monotonically increasing")
    if bh_curve[0, 0] >= 0 or bh_curve[0, 1] < 0 or bh_curve[-1, 0] < 0 or bh_curve[-1, 1] <= 0:
        raise ValueError("Bad BH curve: second quadrant not fully covered")

    coef = copy.copy(ja.COEF_INITIAL)

    sol = ja.solve(coef, H_step=1.0, dM_dH_saturation_threshold=1.0, H_magnitude_limit=1e6)

    _logger.info(f"Solution contains fragments of size: {[len(x) for x in sol.H_M_B_segments]}")

    visualize(sol, bh_curve)


_logger = getLogger(__name__)


if __name__ == "__main__":
    main()
