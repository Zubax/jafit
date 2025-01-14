# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

from logging import getLogger
from pathlib import Path
import numpy as np
import numpy.typing as npt
from .mag import mu_0, HysteresisLoop, hm_to_hb


def load(path_or_text: Path | str) -> HysteresisLoop:
    r"""
    Reads a BH curve from the tab-separated (TSV) file, where the first column is H and the second column is B.
    There may or may not be a header row; this is detected automatically.
    The values can be ascending or descending with respect to H; they will be sorted in the H-ascending order.
    The input table is expected to contain either:

    - The demagnetization curve alone. The H ordering can be arbitrary; its derivative does not change the sign.

    - The full hysteresis loop including the magnetization and demagnetization segments in an arbitrary order.
      The boundary between the mag/demag curves is determined from the sign change in the H-derivative.
      Again, the H-ordering can be arbitrary.

    Note that the returned data is actually an M(H) curve; the transformation is done internally.
    The function can be trivially extended to load M(H), J(H) [aka B_i(H)] curves as well if needed.

    >>> hm_mag_demag = "\n".join(
    ...     ("H\tB", "-1.1\t-1.9", "0.1\t0.9", "1.1\t1.9", "2.1\t2.9", "1.2\t1.8", "0.2\t0.8", "-1.2\t-1.8"))
    >>> loop = load(hm_mag_demag)
    >>> loop.descending[:,0].tolist()
    [-1.2, 0.2, 1.2, 2.1]
    >>> loop.ascending[:,0].tolist()
    [-1.1, 0.1, 1.1, 2.1]

    >>> hm_mag_demag = "\n".join(
    ...     ("2.1\t2.9", "1.2\t1.8", "0.2\t0.8", "-1.2\t-1.8", "-1.1\t-1.9", "0.1\t0.9", "1.1\t1.9"))
    >>> loop = load(hm_mag_demag)
    >>> loop.descending[:,0].tolist()
    [-1.2, 0.2, 1.2, 2.1]
    >>> loop.ascending[:,0].tolist()
    [-1.2, -1.1, 0.1, 1.1]

    >>> loop = load("\n".join(("2.1\t2.9", "1.2\t1.8", "0.2\t0.8", "-1.2\t-1.8")))
    >>> loop.descending[:,0].tolist()
    [-1.2, 0.2, 1.2, 2.1]
    >>> loop.ascending[:,0].tolist()
    []
    """
    if isinstance(path_or_text, Path):
        path_or_text = path_or_text.read_text()
    if isinstance(path_or_text, str):
        lines = path_or_text.splitlines()
    else:
        raise TypeError(f"Invalid argument type: {type(path_or_text).__name__}")
    try:
        [float(x) for x in lines[0].split()]
    except ValueError:
        _logger.info("Skipping the first line of the input file, assuming it is the header: %r", lines[0])
        lines = lines[1:]
    m = np.array([[float(x) for x in line.split()] for line in lines], dtype=np.float64)
    if len(m.shape) != 2 or m.shape[1] != 2 or m.shape[0] < 2:
        raise ValueError(f"Invalid data shape: {m.shape}")

    def finalize(c: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        diff = np.diff(c[:, 0])
        if np.all(diff > 0):
            pass
        elif np.all(diff < 0):
            c = c[::-1]
        else:
            assert False, "Not supposed to get here by design; see the place of invocation"
        c[:, 1] = c[:, 1] / mu_0 - c[:, 0]
        return c

    d = np.diff(m[:, 0])
    d = np.concatenate(([d[0]], d))
    sign_changes_after_indices = d[1:] * d[:-1] < 0
    sign_changes = np.sum(sign_changes_after_indices)
    if sign_changes == 0:
        m = finalize(m)
        return HysteresisLoop(descending=m, ascending=np.empty((0, 2), dtype=np.float64))
    if sign_changes == 1:
        sign_changes_after = np.flatnonzero(sign_changes_after_indices)[0]
        a, b = m[sign_changes_after:], m[: sign_changes_after + 1]
        if a[0, 0] < b[0, 0]:  # Ensure the H-descending curve is stored in a
            a, b = b, a
        return HysteresisLoop(descending=finalize(a), ascending=finalize(b))

    raise ValueError(
        f"The input file appears to contain more than two branches of the hysteresis loop. "
        f"The H-column can change the sign of its derivative either zero times or once; "
        f"in this dataset this happens {sign_changes} times."
    )


def save(file_path: Path, data: HysteresisLoop | npt.NDArray[np.float64]) -> None:
    """
    Saves the given hysteresis loop to a tab-separated (TSV) file containing two columns: H and B.
    """
    if isinstance(data, HysteresisLoop):
        return save(file_path, np.vstack((data.descending, data.ascending)))

    def table(hm: npt.NDArray[np.float64]) -> str:
        hb = hm_to_hb(hm)
        return "\n".join(f"{h:+015.6f}\t{b:+015.12f}" for h, b in hb) + "\n"

    assert isinstance(data, np.ndarray)
    rows, cols = data.shape
    if cols != 2:
        raise ValueError(f"Invalid shape of the M(H) curve: {data.shape}")
    if rows == 0:
        raise ValueError("No data to save")
    text = "H [ampere/meter]\tB [tesla]\n" + table(data)

    if not file_path.parent.exists():
        file_path.parent.mkdir(parents=True)
    file_path.write_text(text)


_logger = getLogger(__name__)
