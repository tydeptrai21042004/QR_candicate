from __future__ import annotations

from dataclasses import asdict
from time import perf_counter
from typing import Any

import numpy as np

from ..attacks import apply_attack
from ..core.interfaces import WatermarkMethod
from ..core.types import BenchmarkRecord
from .metrics import ber, nc, psnr, ssim


def evaluate_once(
    method: WatermarkMethod,
    host: np.ndarray,
    watermark: np.ndarray,
    *,
    key: bytes,
    host_name: str,
    watermark_name: str,
    attack_name: str = "clean",
    attack_params: dict[str, Any] | None = None,
    seed: int | None = None,
) -> tuple[BenchmarkRecord, dict[str, Any]]:
    attack_params = dict(attack_params or {})
    if seed is not None and attack_name in {"gaussian_noise", "salt_pepper", "speckle_noise", "occlusion"}:
        attack_params.setdefault("seed", seed)

    t0 = perf_counter()
    emb = method.embed(host, watermark, key=key)
    embed_s = perf_counter() - t0

    attacked = apply_attack(attack_name, emb.image, **attack_params)

    t0 = perf_counter()
    ext = method.extract(attacked, key=key, side_info=emb.side_info, watermark_shape=watermark.shape[:2])
    extract_s = perf_counter() - t0

    record = BenchmarkRecord(
        method=method.name,
        host=host_name,
        watermark=watermark_name,
        attack=attack_name,
        attack_params=str(attack_params),
        seed=seed,
        psnr=psnr(host, attacked),
        ssim=ssim(host, attacked),
        nc=nc(watermark, ext.watermark),
        ber=ber(watermark, ext.watermark),
        embed_seconds=float(embed_s),
        extract_seconds=float(extract_s),
        confidence=ext.confidence,
    )
    extras = {
        "watermarked": emb.image,
        "attacked": attacked,
        "extracted": ext.watermark,
        "embedding_metadata": emb.metadata,
        "extraction_metadata": ext.metadata,
        "side_info": emb.side_info,
    }
    return record, extras
