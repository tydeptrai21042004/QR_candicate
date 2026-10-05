from __future__ import annotations

from functools import lru_cache
import hashlib
import hmac

import numpy as np


@lru_cache(maxsize=32)
def _keyed_permutation_cached(length: int, key: bytes, domain: bytes) -> tuple[int, ...]:
    """Session-cacheable deterministic HMAC-SHA256 permutation."""
    if not key:
        raise ValueError("A non-empty key is required")
    records: list[tuple[bytes, int]] = []
    for i in range(int(length)):
        msg = domain + i.to_bytes(8, "big")
        records.append((hmac.new(key, msg, hashlib.sha256).digest(), i))
    records.sort(key=lambda x: x[0])
    return tuple(i for _, i in records)


def keyed_permutation(length: int, key: bytes, domain: bytes = b"qrwatermark:block-order:v2") -> np.ndarray:
    cached = _keyed_permutation_cached(int(length), bytes(key), bytes(domain))
    return np.fromiter(cached, dtype=np.int64, count=int(length))


@lru_cache(maxsize=64)
def _selected_block_positions_cached(
    shape: tuple[int, int], block_size: int, count: int, key: bytes
) -> tuple[tuple[int, int], ...]:
    h, w = int(shape[0]), int(shape[1])
    bs = int(block_size)
    rows = h // bs
    cols = w // bs
    total = rows * cols
    if int(count) > total:
        raise ValueError(f"Capacity insufficient: requested {count} blocks, available {total}")
    perm = _keyed_permutation_cached(total, bytes(key), b"qrwatermark:block-order:v2")[: int(count)]
    return tuple(((idx // cols) * bs, (idx % cols) * bs) for idx in perm)


@lru_cache(maxsize=64)
def _selected_block_arrays_cached(
    shape: tuple[int, int], block_size: int, count: int, key: bytes
) -> tuple[np.ndarray, np.ndarray]:
    """Cached row/column arrays for the real-time datapath.

    This is the same HMAC permutation as ``selected_block_positions`` but
    avoids rebuilding 4096 Python tuples and converting them back to NumPy on
    every frame.  Arrays are read-only so cached session state cannot be
    accidentally mutated by callers.
    """
    h, w = int(shape[0]), int(shape[1])
    bs = int(block_size)
    rows = h // bs
    cols = w // bs
    total = rows * cols
    if int(count) > total:
        raise ValueError(f"Capacity insufficient: requested {count} blocks, available {total}")
    perm = np.fromiter(
        _keyed_permutation_cached(total, bytes(key), b"qrwatermark:block-order:v2")[: int(count)],
        dtype=np.int64,
        count=int(count),
    )
    rr = ((perm // cols) * bs).astype(np.intp, copy=False)
    cc = ((perm % cols) * bs).astype(np.intp, copy=False)
    rr.flags.writeable = False
    cc.flags.writeable = False
    return rr, cc


def selected_block_positions(shape: tuple[int, int], block_size: int, count: int, key: bytes) -> list[tuple[int, int]]:
    return list(_selected_block_positions_cached(tuple(map(int, shape)), int(block_size), int(count), bytes(key)))


def selected_block_arrays(
    shape: tuple[int, int], block_size: int, count: int, key: bytes
) -> tuple[np.ndarray, np.ndarray]:
    """Return cached read-only row/column vectors for selected blocks."""
    return _selected_block_arrays_cached(tuple(map(int, shape)), int(block_size), int(count), bytes(key))


def clear_permutation_cache() -> None:
    _selected_block_arrays_cached.cache_clear()
    _selected_block_positions_cached.cache_clear()
    _keyed_permutation_cached.cache_clear()
