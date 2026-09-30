from __future__ import annotations

import hashlib
import hmac
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

MAGIC = "CCQRSIDE2"
LEGACY_MAGIC = "CCQRSIDE1"


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


def _pack_mask(mask: np.ndarray) -> np.ndarray:
    return np.packbits(np.asarray(mask,dtype=np.uint8).ravel(),bitorder="little")


def _payload(side_info: dict[str, Any]) -> bytes:
    meta=dict(side_info["metadata"])
    out=_canonical_metadata(meta)+np.asarray(side_info["packed_period_codes"],dtype=np.uint8).tobytes()
    if "packed_certified_mask" in side_info:
        out+=np.asarray(side_info["packed_certified_mask"],dtype=np.uint8).tobytes()
    return out


def build_side_info(period_indices: np.ndarray, metadata: dict[str, Any], key: bytes, certified_mask: np.ndarray | None=None) -> dict[str, Any]:
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
    side={"metadata":meta,"packed_period_codes":packed}
    if certified_mask is not None:
        mask=np.asarray(certified_mask,dtype=np.uint8).ravel()
        if mask.size!=codes.size: raise ValueError("certified_mask length must match period codes")
        meta["certified_mask_count"]=int(mask.size)
        side["packed_certified_mask"]=_pack_mask(mask)
    side["tag"]=hmac.new(key,_payload(side),hashlib.sha256).hexdigest()
    return side


def _verify(side_info: dict[str, Any],key:bytes)->dict[str,Any]:
    meta=dict(side_info["metadata"])
    if meta.get("magic") not in {MAGIC,LEGACY_MAGIC}:
        raise ValueError("Invalid side-information magic/version")
    expected=hmac.new(key,_payload(side_info),hashlib.sha256).hexdigest()
    if not hmac.compare_digest(str(side_info["tag"]),expected):
        raise ValueError("Side-information authentication failed")
    return meta


def unpack_period_indices(side_info: dict[str, Any], key: bytes) -> tuple[np.ndarray, dict[str, Any]]:
    meta=_verify(side_info,key)
    packed = np.asarray(side_info["packed_period_codes"], dtype=np.uint8).ravel()
    count = int(meta["period_code_count"]); width = int(meta["period_code_bits"])
    codes = _unpack_codes(packed, count, width)
    if np.any(codes >= int(meta["period_count"])):
        raise ValueError("Corrupt period code in side information")
    return codes, meta


def unpack_certified_mask(side_info:dict[str,Any],key:bytes)->np.ndarray:
    meta=_verify(side_info,key)
    count=int(meta.get("certified_mask_count",meta.get("period_code_count",0)))
    if "packed_certified_mask" not in side_info:
        return np.zeros(count,dtype=bool)
    bits=np.unpackbits(np.asarray(side_info["packed_certified_mask"],dtype=np.uint8),bitorder="little")[:count]
    return bits.astype(bool)


def side_info_overhead_bits(side_info:dict[str,Any])->dict[str,int]:
    meta=dict(side_info["metadata"])
    period_bits=int(meta.get("period_code_count",0))*int(meta.get("period_code_bits",0))
    cert_bits=int(meta.get("certified_mask_count",0)) if "packed_certified_mask" in side_info else 0
    metadata_bits=8*len(_canonical_metadata(meta))
    authentication_bits=256
    packed_storage_bits=8*(len(np.asarray(side_info["packed_period_codes"],dtype=np.uint8).ravel())+len(np.asarray(side_info.get("packed_certified_mask",[]),dtype=np.uint8).ravel()))
    total_serialized_bits=metadata_bits+packed_storage_bits+authentication_bits
    return {
        "period_code_bits":period_bits,
        "certificate_mask_bits":cert_bits,
        "authentication_bits":authentication_bits,
        "metadata_bits":metadata_bits,
        "packed_storage_bits":packed_storage_bits,
        "total_serialized_bits":total_serialized_bits,
    }


def save_side_info(path: str | Path, side_info: dict[str, Any]) -> None:
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    kwargs={
        "metadata":np.asarray(json.dumps(side_info["metadata"],sort_keys=True)),
        "packed_period_codes":np.asarray(side_info["packed_period_codes"],dtype=np.uint8),
        "tag":np.asarray(str(side_info["tag"])),
    }
    if "packed_certified_mask" in side_info:
        kwargs["packed_certified_mask"]=np.asarray(side_info["packed_certified_mask"],dtype=np.uint8)
    np.savez_compressed(path,**kwargs)


def load_side_info(path: str | Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as z:
        out={
            "metadata": json.loads(str(z["metadata"].item())),
            "packed_period_codes": z["packed_period_codes"].astype(np.uint8),
            "tag": str(z["tag"].item()),
        }
        if "packed_certified_mask" in z.files:
            out["packed_certified_mask"]=z["packed_certified_mask"].astype(np.uint8)
        return out
