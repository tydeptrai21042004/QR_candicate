from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter


@contextmanager
def timer():
    box = {"seconds": None}
    t0 = perf_counter()
    try:
        yield box
    finally:
        box["seconds"] = perf_counter() - t0
