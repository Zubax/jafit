# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

from logging import getLogger
from pathlib import Path
import numpy as np
import numpy.typing as npt
from .mag import mu_0, HysteresisLoop, hm_to_hb


def load(
    path_or_text: Path | str,
    *,
    possible_column_separators: str = "\t,",
    kind: str = "B(H)",
) -> HysteresisLoop:
    r"""
    Reads a BH curve from a tab-separated (TSV) or comma-separated (CSV) file,
    where the first column is H and the second column is B.
    There may or may not be a header row; this is detected automatically.
    The values can be ascending or descending with respect to H; they will be sorted in the H-ascending order.
    The input table is expected to contain either:

    - The demagnetization curve alone. The H ordering can be arbitrary; its derivative does not change the sign.

    - The full hysteresis loop including the magnetization and demagnetization segments in an arbitrary order.
      The boundary between the mag/demag curves is determined from the sign change in the H-derivative.
      Again, the H-ordering can be arbitrary.

    Note that the returned data is actually an M(H) curve; the transformation is done internally.
    The function can be trivially extended to load M(H), J(H) [aka B_i(H)] curves as well if needed.

    The column separator can be one of the specified characters; each is tried in order until a match is found.

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

    >>> loop = load("\n".join(("2.1,2.9", "1.2, 1.8", "0.2 ,0.8", "-1.2 , -1.8")))
    >>> loop.descending[:,0].tolist()
    [-1.2, 0.2, 1.2, 2.1]
    """
    if isinstance(path_or_text, Path):
        path_or_text = path_or_text.read_text()
    if isinstance(path_or_text, str):
        lines = path_or_text.strip().splitlines()
    else:
        raise TypeError(f"Invalid argument type: {type(path_or_text).__name__}")

    col_sep = ""
    for sep in possible_column_separators:
        if sep in lines[0]:
            col_sep = sep
            break
    if not col_sep:
        raise ValueError(f"Cannot detect the column separator in the first line of the input file: {lines[0]!r}")

    try:
        [float(x.strip()) for x in lines[0].split(col_sep)]
    except ValueError:
        _logger.info("Skipping the first line of the input file, assuming it is the header: %r", lines[0])
        lines = lines[1:]
    m = np.array([[float(x.strip()) for x in line.split(col_sep)] for line in lines], dtype=np.float64)
    if len(m.shape) != 2 or m.shape[1] != 2 or m.shape[0] < 2:
        raise ValueError(f"Invalid data shape: {m.shape}")

    # Remove repeated H values. This happens in some datasets.
    m_original = m
    _, m_unique_idx = np.unique(m[:, 0], return_index=True)  # m_unique_idx holds the first occurrence of each H value
    m_unique_idx = np.sort(m_unique_idx)  # restore the original ordering
    m = m[m_unique_idx]
    if len(m) < len(m_original):
        _logger.info(
            "Removed %d points with the same H values: was %d points, now %d points",
            len(m_original) - len(m),
            len(m_original),
            len(m),
        )

    # Handle the easy case where the H-sign is not noisy: clean mag/demag curves separated by a diff sign change.
    d = np.diff(m[:, 0])
    d = np.concatenate(([d[0]], d))
    sign_changes_after_indices = d[1:] * d[:-1] < 0
    sign_changes = np.sum(sign_changes_after_indices)
    if sign_changes == 0:
        m = _finalize_branch(m, kind)
        return HysteresisLoop(descending=m, ascending=np.empty((0, 2), dtype=np.float64))
    if sign_changes == 1:
        sign_changes_after = np.flatnonzero(sign_changes_after_indices)[0]
        a, b = m[sign_changes_after:], m[: sign_changes_after + 1]
        if a[0, 0] < b[0, 0]:  # Ensure the H-descending curve is stored in a
            a, b = b, a
        return HysteresisLoop(descending=_finalize_branch(a, kind), ascending=_finalize_branch(b, kind))

    # The harder case.
    _logger.debug("Multiple sign changes detected in the input curve; assuming the input data is sign-noisy")
    H_start = m[0, 0]
    H_end = m[-1, 0]
    if (H_start > 0) == (H_end > 0):  # Assume we have the full loop because the curve returns to the same sign
        H_flip_index = np.argmin(m[:, 0]) if H_start > 0 else np.argmax(m[:, 0])
        H_flip = m[H_flip_index, 0]
        _logger.debug("H_start=%+f, H_flip=%+f, H_end=%+f", H_start, H_flip, H_end)
        if (H_start > 0) == (H_flip > 0):
            raise ValueError("Unsupported input data: the full loop does not cross the vertical axis")
        if H_start > 0:
            assert H_flip < 0
            dsc, asc = m[: H_flip_index + 1], m[H_flip_index:]
        else:
            assert H_flip > 0
            dsc, asc = m[H_flip_index:], m[: H_flip_index + 1]
        return HysteresisLoop(
            descending=_finalize_branch(dsc, kind),
            ascending=_finalize_branch(asc, kind),
        )

    raise ValueError(
        "Currently, there is no support for loading single-branch curves with noisy sign changes,"
        " although it can be added easily if needed."
    )


def _finalize_branch(c: npt.NDArray[np.float64], kind: str) -> npt.NDArray[np.float64]:
    """
    Ensure the correct ordering and convert the units into M(H).
    """
    if np.all(np.diff(c[:, 0]) > 0):
        pass
    elif np.all(np.diff(c[:, 0]) < 0):
        c = c[::-1]
    else:
        if c[0, 0] > c[-1, 0]:
            c = c[::-1]
        # Denoise: remove entries where H-field goes backwards or doesn't change.
        # We cannot simply use a mask=np.diff(c[:,0]>0) because it breaks in sequences like H=[1,3,5,7,4,6,8],
        # where after the mask application we get H=[1,3,5,7,6,8].
        keep, current_max = [], -np.inf
        for x, _ in c:
            if x > current_max:
                keep.append(True)
                current_max = x
            else:
                keep.append(False)
        c_mew = c[np.array(keep)]
        _logger.debug(
            "Removed %d points with the same or decreasing H values: was %d points, now %d points",
            len(c) - len(c_mew),
            len(c),
            len(c_mew),
        )
        c = c_mew

    assert np.all(np.diff(c[:, 0]) > 0), f"{[float(x) for x in c[:, 0]]}"

    kind = kind.upper().strip().replace(" ", "")
    if kind == "B(H)":
        # Note that we cannot simply do  c[:,1]=c[:,1]/mu_0-c[:,0]  because it damages the memory
        c = np.column_stack((c[:, 0], c[:, 1] / mu_0 - c[:, 0]))
    elif kind == "M(H)":
        pass
    else:
        raise ValueError(f"Unsupported curve kind: {kind!r}")

    if np.abs(c).max() > 10e6:
        _logger.warning("The loaded curve appears to be in the wrong units: values seem too large:\n%s", c)
    return c


def save(file_path: Path, data: HysteresisLoop | npt.NDArray[np.float64]) -> None:
    """
    Saves the given hysteresis loop to a tab-separated (TSV) file containing two columns: H and B.
    """
    if isinstance(data, HysteresisLoop):
        # Should we reverse one of the curves to make H more contiguous?
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
