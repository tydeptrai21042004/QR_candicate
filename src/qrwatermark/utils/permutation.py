from __future__ import annotations

import hashlib
import hmac

import numpy as np


def keyed_permutation(length: int, key: bytes, domain: bytes = b"qrwatermark:block-order:v2") -> np.ndarray:
    """Deterministic HMAC-SHA256 permutation; independent of floating-point chaotic maps."""
    if not key:
        raise ValueError("A non-empty key is required")
    records: list[tuple[bytes, int]] = []
    for i in range(length):
        msg = domain + i.to_bytes(8, "big")
        records.append((hmac.new(key, msg, hashlib.sha256).digest(), i))
    records.sort(key=lambda x: x[0])
    return np.fromiter((i for _, i in records), dtype=np.int64, count=length)


def selected_block_positions(shape: tuple[int, int], block_size: int, count: int, key: bytes) -> list[tuple[int, int]]:
    h, w = shape
    rows = h // block_size
    cols = w // block_size
    total = rows * cols
    if count > total:
        raise ValueError(f"Capacity insufficient: requested {count} blocks, available {total}")
    perm = keyed_permutation(total, key)[:count]
    return [((int(idx) // cols) * block_size, (int(idx) % cols) * block_size) for idx in perm]
