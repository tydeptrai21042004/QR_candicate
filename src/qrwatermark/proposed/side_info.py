from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = "CCQRSIDE1"


def _canonical_metadata(metadata: dict[str, Any]) -> bytes:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _bit_width(n_symbols: int) -> int:
    if n_symbols < 1:
        raise ValueError("n_symbols must be positive")
    return max(1, int(math.ceil(math.log2(n_symbols))))


def _pack_codes(codes: np.ndarray, bit_width: int) -> np.ndarray:
    codes = np.asarray(codes, dtype=np.uint16).ravel()
    bits = np.empty(codes.size * bit_width, dtype=np.uint8)
    for j in range(bit_width):
        bits[j::bit_width] = ((codes >> j) & 1).astype(np.uint8)
    return np.packbits(bits, bitorder="little")


def _unpack_codes(packed: np.ndarray, count: int, bit_width: int) -> np.ndarray:
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8), bitorder="little")[: count * bit_width]
    matrix = bits.reshape(count, bit_width).astype(np.uint16)
    values = np.zeros(count, dtype=np.uint16)
    for j in range(bit_width):
        values |= matrix[:, j] << j
    return values.astype(np.uint8)


def build_side_info(period_indices: np.ndarray, metadata: dict[str, Any], key: bytes) -> dict[str, Any]:
    codes = np.asarray(period_indices, dtype=np.uint8).ravel()
    period_count = int(metadata["period_count"])
    if np.any(codes >= period_count):
        raise ValueError("period index exceeds configured period table")
    width = _bit_width(period_count)
    packed = _pack_codes(codes, width)
    meta = dict(metadata)
    meta["magic"] = MAGIC
    meta["period_code_count"] = int(codes.size)
    meta["period_code_bits"] = int(width)
    payload = _canonical_metadata(meta) + packed.tobytes()
    tag = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return {"metadata": meta, "packed_period_codes": packed, "tag": tag}


def unpack_period_indices(side_info: dict[str, Any], key: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    meta = dict(side_info["metadata"])
    if meta.get("magic") != MAGIC:
        raise ValueError("Invalid side-information magic/version")
    packed = np.asarray(side_info["packed_period_codes"], dtype=np.uint8).ravel()
    payload = _canonical_metadata(meta) + packed.tobytes()
    expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(str(side_info["tag"]), expected):
        raise ValueError("Side-information authentication failed")
    count = int(meta["period_code_count"])
    width = int(meta["period_code_bits"])
    codes = _unpack_codes(packed, count, width)
    if np.any(codes >= int(meta["period_count"])):
        raise ValueError("Corrupt period code in side information")
    return codes, meta


def save_side_info(path: str | Path, side_info: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(side_info["metadata"], sort_keys=True)),
        packed_period_codes=np.asarray(side_info["packed_period_codes"], dtype=np.uint8),
        tag=np.asarray(str(side_info["tag"])),
    )


def load_side_info(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {
            "metadata": json.loads(str(z["metadata"].item())),
            "packed_period_codes": z["packed_period_codes"].astype(np.uint8),
            "tag": str(z["tag"].item()),
        }
