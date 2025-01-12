# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

from logging import getLogger
import numpy as np
import numpy.typing as npt
import matplotlib

matplotlib.use("Agg")  # Choose the noninteractive backend; this has to be done before importing pyplot
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

mu_0 = 1.2566370614359173e-6  # Vacuum permeability [henry/meter]


def plot(
    virgin: npt.NDArray[np.float64],
    major_descending: npt.NDArray[np.float64],
    major_ascending: npt.NDArray[np.float64],
    output_file_name: str,
    bh_curve_ref: npt.NDArray[np.float64] | None = None,
    max_points: float = 1e4,
) -> None:
    fig, ax_b = plt.subplots(1, 1, figsize=(14, 10), sharex="all")  # type: ignore
    try:

        def plot_hmb(fragment: npt.NDArray[np.float64], prefix: str) -> None:
            n_points = fragment.shape[0]
            if n_points > max_points:  # Select `max_points` evenly spaced indices from the fragment
                indices = np.round(np.linspace(0, n_points - 1, int(max_points))).astype(int)
                fragment = fragment[indices, :]
            H_vals, M_vals, B_vals = fragment[:, 0], fragment[:, 1], fragment[:, 2]
            J_vals = M_vals * mu_0
            ax_b.plot(H_vals, J_vals, label=f"{prefix} polarization J")
            ax_b.plot(H_vals, B_vals, label=f"{prefix} flux density B")

        plot_hmb(virgin, "Virgin")
        plot_hmb(major_descending, "Major descending")
        plot_hmb(major_ascending, "Major ascending")

        # Plot the reference BH curve
        if bh_curve_ref is not None:
            ax_b.scatter(bh_curve_ref[:, 0], bh_curve_ref[:, 1], marker=".", label="Reference BH curve")

        # Configure B(H)|J(H) subplot
        ax_b.set_title("Flux density and polarization vs. external field")
        ax_b.set_xlabel("H [ampere/meter]")
        ax_b.set_ylabel("B|J [tesla]")
        ax_b.legend()
        ax_b.xaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax_b.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax_b.grid(which="major", color="gray", linestyle="-", alpha=0.7)
        ax_b.grid(which="minor", color="gray", linestyle=":", alpha=0.5)

        # Show the plot
        plt.tight_layout()
        _logger.debug(f"Saving the plot to {output_file_name!r}")
        plt.savefig(output_file_name)
    finally:
        plt.close(fig)


_logger = getLogger(__name__)
