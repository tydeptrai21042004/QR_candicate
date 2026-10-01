from __future__ import annotations

from functools import lru_cache
import hashlib
import hmac

import numpy as np


@lru_cache(maxsize=32)
def _keyed_permutation_cached(length: int, key: bytes, domain: bytes) -> tuple[int, ...]:
    """Session-cacheable deterministic HMAC-SHA256 permutation.

    The permutation is a control-plane object: for a fixed image geometry,
    key, and domain it never changes between video frames.  Caching it removes
    the 65k HMAC+sort operation from the edge per-frame datapath without
    changing a single selected block.
    """
    if not key:
        raise ValueError("A non-empty key is required")
    records: list[tuple[bytes, int]] = []
    for i in range(int(length)):
        msg = domain + i.to_bytes(8, "big")
        records.append((hmac.new(key, msg, hashlib.sha256).digest(), i))
    records.sort(key=lambda x: x[0])
    return tuple(i for _, i in records)


def keyed_permutation(length: int, key: bytes, domain: bytes = b"qrwatermark:block-order:v2") -> np.ndarray:
    """Deterministic HMAC-SHA256 permutation; identical to the legacy result.

    A fresh NumPy array is returned so callers cannot mutate the cached session
    state.  On an edge deployment the cached tuple maps naturally to descriptor
    ROM/BRAM generated once when the key/session is established.
    """
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


def selected_block_positions(shape: tuple[int, int], block_size: int, count: int, key: bytes) -> list[tuple[int, int]]:
    """Selected positions with a session cache for real-time edge reuse."""
    return list(_selected_block_positions_cached(tuple(map(int, shape)), int(block_size), int(count), bytes(key)))


def clear_permutation_cache() -> None:
    """Clear cached control-plane descriptors (useful when rotating keys)."""
    _selected_block_positions_cached.cache_clear()
    _keyed_permutation_cached.cache_clear()
