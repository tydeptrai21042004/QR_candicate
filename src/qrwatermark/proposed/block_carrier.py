"""Recovery carrier mapper for repository snapshots without fused_convqr_v6.

Caution: this mapper has a new deterministic keyed permutation. It is NOT
known to be bit-identical to the missing original FCQR-v6 carrier mapper.
All historical numerical comparisons must be rerun after applying it.
"""
from __future__ import annotations
import hashlib
import numpy as np

def keyed_blocks_v6(image_shape, count, key):
    h, w = int(image_shape[0]), int(image_shape[1])
    rows, cols = h // 4, w // 4
    capacity = rows * cols
    n = int(count)
    if n < 0 or n > capacity:
        raise ValueError(f"requested {n} 4x4 blocks; available {capacity}")
    if not isinstance(key, bytes):
        raise TypeError("key must be bytes")
    payload = (b"qrwatermark-recovered-block-carrier-v1\0" +
               len(key).to_bytes(4, "big") + key +
               h.to_bytes(8, "big") + w.to_bytes(8, "big"))
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")
    rng = np.random.Generator(np.random.PCG64(seed))
    ix = rng.permutation(capacity)[:n]
    return (ix // cols * 4).astype(np.intp), (ix % cols * 4).astype(np.intp)

def _gather_blocks(channel, rr, cc):
    x = np.asarray(channel)
    if x.ndim != 2:
        raise ValueError("expected 2D channel")
    rr = np.asarray(rr, dtype=np.intp).ravel()
    cc = np.asarray(cc, dtype=np.intp).ravel()
    if rr.shape != cc.shape:
        raise ValueError("row/column index mismatch")
    off = np.arange(4, dtype=np.intp)
    return x[rr[:, None, None]+off[None, :, None],
             cc[:, None, None]+off[None, None, :]].copy()
