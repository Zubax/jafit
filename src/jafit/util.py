# Copyright (C) 2025 Pavel Kirienko <pavel.kirienko@zubax.com>

from typing import Any, Callable

try:
    from numba import jit, njit
except ImportError:

    def jit(**_: Any) -> Callable[[Callable], Callable]:
        return lambda f: f

    njit = jit
