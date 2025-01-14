# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

from __future__ import annotations
from typing import Any
import time
import dataclasses
from logging import getLogger
import numpy as np
import numpy.typing as npt
from .util import njit

mu_0 = 1.2566370614359173e-6  # Vacuum permeability [henry/meter]


@dataclasses.dataclass(frozen=True)
class HysteresisLoop:
    """
    Each array contains n rows of 2 elements: H-field and M-field.
    All curves are sorted in ascending order of the H-field.
    The decision to use M and H values is made intentionally because by having them use the same units [A/m],
    we can simplify loss function computation (no need to renormalize) and some other operations.
    Either curve (but never both) may be empty if no such data is provided.
    """

    descending: npt.NDArray[np.float64]
    ascending: npt.NDArray[np.float64]

    @property
    def H_range(self) -> tuple[float, float]:
        if len(self.descending) == 0:
            return float(self.ascending[0, 0]), float(self.ascending[-1, 0])
        if len(self.ascending) == 0:
            return float(self.descending[0, 0]), float(self.descending[-1, 0])
        assert self.descending[0, 0] < self.descending[-1, 0]
        assert self.ascending[0, 0] < self.ascending[-1, 0]
        lo = min(float(self.descending[0, 0]), float(self.ascending[0, 0]))
        hi = max(float(self.descending[-1, 0]), float(self.ascending[-1, 0]))
        return lo, hi

    @property
    def is_full(self) -> bool:
        return len(self.descending) > 0 and len(self.ascending) > 0

    def balance(self) -> HysteresisLoop:
        """
        Returns a new HysteresisLoop object where both curves are made symmetric around the origin.
        This is intended to reduce the integration error when the curves are not perfectly symmetric.
        """
        started_at = time.monotonic()
        if not self.is_full:
            raise ValueError("Cannot balance the major loop because one of the curves is missing")
        if max(len(self.descending), len(self.ascending)) / min(len(self.descending), len(self.ascending)) >= 10:
            _logger.warning(
                "HysteresisLoop: Balancing: the curves have significantly different lengths: desc=%d, asc=%d",
                len(self.descending),
                len(self.ascending),
            )

        # Prepare the curves such that they are both in the H-ascending order and have the same polarity.
        # There is one caveat: the sample density may vary in the beginning and at the end of the curves
        # due to the automatic step width adjustment in the ODE solver.
        # If the starting step size is small, the curve may have a higher density at the beginning than at the end.
        # The descending curve will have a higher sample density in the upper-right part of the plot, and vice versa.
        # After one of the curves is mirrored around the origin, they both will have a greater number of samples on
        # one side, which causes the mean curve to be heavily disbalanced as well.
        # This is something to be aware of, but it is probably not a problem in itself.
        dsc = self.descending
        asc = -self.ascending[::-1]  # Mirror around the origin and keep the H-ascending order.
        assert dsc[0, 0] < dsc[-1, 0] and asc[0, 0] < asc[-1, 0]

        # Ensure the final range is covered by both curves; otherwise, interpolation will not be possible.
        # Note that this can only be done AFTER the mirroring, so we cannot use the H_range property here.
        H_lo = max(float(dsc[0, 0]), float(asc[0, 0]))
        H_hi = min(float(dsc[-1, 0]), float(asc[-1, 0]))
        assert H_lo < H_hi

        # Merge all H values from both curves and sort them in the ascending order, removing values too close together.
        H = np.concatenate((dsc[:, 0], asc[:, 0]))
        H = np.unique(H[(H >= H_lo) & (H <= H_hi)])
        H = H[np.concatenate(([True], np.diff(H) > 1e-9))]
        assert np.all(np.diff(H) > 0)
        assert H.min() >= H_lo and H.max() <= H_hi
        assert len(H) >= len(dsc) and len(H) >= len(asc), f"{H=}, {dsc=}, {asc=}"

        # Interpolate both curves at the new H values.
        dsc = np.interp(H, dsc[:, 0], dsc[:, 1])
        asc = np.interp(H, asc[:, 0], asc[:, 1])

        # Compute the mean curve.
        mean = np.column_stack((H, 0.5 * (dsc + asc)))
        mean.setflags(write=False)

        # Log diagnostics, as this is a critical operation.
        # noinspection PyTypeChecker
        def curve_stats(m: npt.NDArray[np.float64]) -> tuple[Any, ...]:
            # noinspection PyTypeChecker
            def lim(i: int) -> tuple[float, float]:
                return m[0, i], m[-1, i]

            return len(m), *lim(0), *lim(1)

        _logger.debug(
            "HysteresisLoop: Balancing result:\n"
            "desc: %7d pts H[%+012.3f,%+012.3f] M[%+012.3f,%+012.3f]\n"
            "asc:  %7d pts H[%+012.3f,%+012.3f] M[%+012.3f,%+012.3f]\n"
            "mean: %7d pts H[%+012.3f,%+012.3f] M[%+012.3f,%+012.3f]\n"
            "removed %d points; elapsed %.0f ms",
            *curve_stats(self.descending),
            *curve_stats(self.ascending),
            *curve_stats(mean),
            len(self.descending) + len(self.ascending) - len(mean),
            (time.monotonic() - started_at) * 1e3,
        )
        return HysteresisLoop(descending=mean, ascending=-mean[::-1])

    def decimate(self, approx_points: int) -> HysteresisLoop:
        """
        Discards every Nth sample from the curves to reduce the size of each to approximately ``approx_points``.
        This is preferred to resampling on a regular lattice because decimation will preserve a higher sample
        density in the high-gradient regions of the curve.
        Curves that contain approximately the same number of points or less will be returned as-is.
        """
        if approx_points < 3:
            raise ValueError(f"Invalid number of points: {approx_points}")
        threshold = 2 * approx_points
        dsc = self.descending
        asc = self.ascending
        if len(dsc) > threshold:
            dsc = dsc[:: len(dsc) // approx_points]
            dsc.setflags(write=False)
        if len(asc) > threshold:
            asc = asc[:: len(asc) // approx_points]
            asc.setflags(write=False)
        return HysteresisLoop(descending=dsc, ascending=asc)

    def __post_init__(self) -> None:
        rows_dsc, cols = self.descending.shape
        if cols != 2:
            raise ValueError("Descending curve out of shape")
        rows_asc, cols = self.ascending.shape
        if cols != 2:
            raise ValueError("Ascending curve out of shape")
        if max(rows_asc, rows_dsc) < 3:
            raise ValueError("Not enough data points in the major loop")
        if not np.all(np.diff(self.descending[:, 0]) > 0):
            raise ValueError("Descending curve is not sorted in ascending order of the H-field")
        if not np.all(np.diff(self.ascending[:, 0]) > 0):
            raise ValueError("Ascending curve is not sorted in ascending order of the H-field")

    def __str__(self) -> str:
        def curve_stats(c: npt.NDArray[np.float64]) -> str:
            if len(c) == 0:
                return "(pts=0)"
            return f"(pts={len(c)}, H=[{c[0, 0]:+012.3f},{c[-1, 0]:+012.3f}], M=[{c[0, 1]:+012.3f},{c[-1, 1]:+012.3f})"

        return (
            f"{type(self).__name__}"
            f"(descending={curve_stats(self.descending)},"
            f" ascending={curve_stats(self.ascending)})"
        )


def extract_H_c_B_r_BH_max(hm: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """
    For any M(H) curve, the computed B_r may have an arbitrary sign, while H_c and BH_max are always positive.
    """
    assert len(hm.shape) == 2 and hm.shape[1] == 2, f"M(H) curve out of shape: {hm.shape}"
    assert np.all(np.diff(hm[:, 0]) > 0)  # interp() requires that the x values are strictly increasing
    bh = np.column_stack((hm[:, 0], mu_0 * hm.sum(axis=1)))
    B_r = np.interp(0, bh[:, 0], bh[:, 1])  # Find B at H=0; trimming the curve is not necessary
    bh = bh[(bh[:, 0] <= 0) & (bh[:, 1] >= 0)]  # Keep only the second quadrant: H<=0, B>=0
    if len(bh) > 0:
        H_c = np.abs(np.min(bh[:, 0]))
        BH_max = -np.min(bh[:, 0] * bh[:, 1])
    else:
        H_c, BH_max = 0, 0
    return float(H_c), float(B_r), float(BH_max)


@njit
def hm_to_hb(hm: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """M(H) => B(H)"""
    return np.column_stack((hm[:, 0], mu_0 * hm.sum(axis=1)))


@njit
def hm_to_hj(hm: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """M(H) => J(H)"""
    return np.column_stack((hm[:, 0], hm[:, 1] * mu_0))


_logger = getLogger(__name__)
