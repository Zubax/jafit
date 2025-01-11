# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

"""
Utilities for handling BH curves.
"""

from logging import getLogger
from pathlib import Path
import numpy as np
import numpy.typing as npt


def extract_H_c_B_r_BH_max_from_major_descending_loop(bh: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    assert len(bh.shape) == 2 and bh.shape[1] == 2, f"BH curve out of shape: {bh.shape}"
    assert np.all(np.diff(bh[:, 0]) > 0)  # interp() requires that the x values are strictly increasing
    B_r = np.interp(0, bh[:, 0], bh[:, 1])  # Find B at H=0; trimming the curve is not necessary
    bh = bh[(bh[:, 0] <= 0) & (bh[:, 1] >= 0)]  # Keep only the second quadrant: H<=0, B>=0
    if len(bh) > 0:
        H_c = np.abs(np.min(bh[:, 0]))
        BH_max = -np.min(bh[:, 0] * bh[:, 1])
    else:
        H_c, BH_max = 0, 0
    return float(H_c), float(B_r), float(BH_max)


def check(bh: npt.NDArray[np.float64]) -> None:
    if not (bh[0, 0] < 0 <= bh[-1, 0]):
        _logger.warning("Suspicious BH curve: H values do not include zero!")
    if not (bh[0, 1] <= 0 < bh[-1, 1]):
        _logger.warning("Suspicious BH curve: B values do not include zero!")
    if not np.all(np.diff(bh[:, 0]) > 0):
        raise ValueError("Bad BH curve: H is not monotonically increasing")
    if len(bh) < 3:
        raise ValueError(f"Bad BH curve: too few data points: {len(bh)}")


def load(file_path: Path) -> npt.NDArray[np.float64]:
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
    bh_data.setflags(write=False)
    return bh_data


_logger = getLogger(__name__)
