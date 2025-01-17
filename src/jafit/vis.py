# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

from logging import getLogger
from pathlib import Path
import enum
import numpy as np
import numpy.typing as npt
import matplotlib

matplotlib.use("Agg")  # Choose the noninteractive backend; this has to be done before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


class Color(enum.Enum):
    black = "black"
    gray = "gray"
    red = "red"
    blue = "blue"


class Style(enum.Enum):
    scatter = enum.auto()
    line = enum.auto()


def plot_hb(
    hb_specs: list[tuple[str, npt.NDArray[np.float64], Style, Color]],
    title: str,
    output_file: str | Path,
    *,
    max_points: float = 5000,
) -> None:
    """
    Plots B(H) data.
    """
    plt.rcParams["font.family"] = "monospace"
    fig, ax_b = plt.subplots(1, 1, figsize=(14, 10), sharex="all")  # type: ignore
    try:

        def trace(hb: npt.NDArray[np.float64], label: str, style: Style, color: str) -> None:
            n_points = hb.shape[0]
            if n_points > max_points * 2:
                indices = np.round(np.linspace(0, n_points - 1, int(max_points))).astype(int)
                hb = hb[indices, :]
            match style:
                case Style.scatter:
                    ax_b.scatter(*hb.T, label=label, color=color, marker=",", s=1)  # type: ignore
                case Style.line:
                    ax_b.plot(*hb.T, label=label, color=color, linestyle="-")

        for name, hb_data, hb_style, hb_color in hb_specs:
            rows, cols = hb_data.shape
            if rows == 0:
                continue
            if cols != 2:
                raise ValueError(f"Invalid data shape: {hb_data.shape}")
            trace(hb_data, name, hb_style, hb_color.value)

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
        _logger.debug(f"Saving the plot to: {str(output_file)!r}")
        plt.savefig(output_file)
    finally:
        plt.close(fig)


_logger = getLogger(__name__)
