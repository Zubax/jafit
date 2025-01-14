# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

from logging import getLogger
from pathlib import Path
import numpy as np
import numpy.typing as npt
import matplotlib

matplotlib.use("Agg")  # Choose the noninteractive backend; this has to be done before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from .mag import hm_to_hb, hm_to_hj


def plot(
    hm_named: dict[str, npt.NDArray[np.float64]],
    title: str,
    output_file: str | Path,
    *,
    max_points: float = 1000,
) -> None:
    plt.rcParams["font.family"] = "monospace"
    fig, ax_b = plt.subplots(1, 1, figsize=(14, 10), sharex="all")  # type: ignore
    try:

        def trace(hm: npt.NDArray[np.float64], label: str) -> None:
            n_points = hm.shape[0]
            if n_points > max_points:  # Select `max_points` evenly spaced indices from the fragment
                indices = np.round(np.linspace(0, n_points - 1, int(max_points))).astype(int)
                hm = hm[indices, :]
            hb = hm_to_hb(hm)
            hj = hm_to_hj(hm)
            ax_b.plot(*hj.T, label=f"J(H) {label}")
            ax_b.plot(*hb.T, label=f"B(H) {label}")

        for name, hm_data in hm_named.items():
            rows, cols = hm_data.shape
            if rows == 0:
                continue
            if cols != 2:
                raise ValueError(f"Invalid shape of the M(H) curve: {hm_data.shape}")
            trace(hm_data, name)

        # Configure B(H)|J(H) subplot
        ax_b.set_title(title)
        ax_b.set_xlabel("H [ampere/meter]")
        ax_b.set_ylabel("B [tesla]")
        ax_b.legend()
        ax_b.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax_b.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax_b.grid(which="major", color="gray", linestyle="-", alpha=0.7)
        ax_b.grid(which="minor", color="gray", linestyle=":", alpha=0.5)

        # Show the plot
        plt.tight_layout()
        if not isinstance(output_file, Path):
            output_file = Path(output_file)
        if not output_file.parent.exists():
            output_file.parent.mkdir(parents=True)
        _logger.debug(f"Saving the plot to: {output_file}")
        plt.savefig(output_file)
    finally:
        plt.close(fig)


_logger = getLogger(__name__)
