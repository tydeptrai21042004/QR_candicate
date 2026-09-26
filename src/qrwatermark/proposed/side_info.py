from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = "QRWMSIDE2"


def _canonical_metadata(metadata: dict[str, Any]) -> bytes:
    return json.dumps(metadata, sort_keys=True, separators=(",", ":")).encode("utf-8")


def build_side_info(modes: np.ndarray, metadata: dict[str, Any], key: bytes) -> dict[str, Any]:
    modes = np.asarray(modes, dtype=np.uint8).ravel()
    if np.any((modes != 0) & (modes != 1)):
        raise ValueError("modes must contain only 0/1")
    packed = np.packbits(modes, bitorder="little")
    meta = dict(metadata)
    meta["magic"] = MAGIC
    meta["mode_count"] = int(modes.size)
    payload = _canonical_metadata(meta) + packed.tobytes()
    tag = hmac.new(key, payload, hashlib.sha256).hexdigest()
    return {"metadata": meta, "packed_modes": packed, "tag": tag}


def unpack_modes(side_info: dict[str, Any], key: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    meta = dict(side_info["metadata"])
    if meta.get("magic") != MAGIC:
        raise ValueError("Invalid side-information magic/version")
    packed = np.asarray(side_info["packed_modes"], dtype=np.uint8).ravel()
    payload = _canonical_metadata(meta) + packed.tobytes()
    expected = hmac.new(key, payload, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(str(side_info["tag"]), expected):
        raise ValueError("Side-information authentication failed")
    count = int(meta["mode_count"])
    modes = np.unpackbits(packed, bitorder="little")[:count].astype(np.uint8)
    return modes, meta


def save_side_info(path: str | Path, side_info: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        metadata=np.asarray(json.dumps(side_info["metadata"], sort_keys=True)),
        packed_modes=np.asarray(side_info["packed_modes"], dtype=np.uint8),
        tag=np.asarray(str(side_info["tag"])),
    )


def load_side_info(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        return {
            "metadata": json.loads(str(z["metadata"].item())),
            "packed_modes": z["packed_modes"].astype(np.uint8),
            "tag": str(z["tag"].item()),
        }
