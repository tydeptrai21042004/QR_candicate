from __future__ import annotations

"""Fully blind extraction for deterministic-carrier QR safe-set QIM."""

import numpy as np

from ..core.config import ProposedConfig
from ..utils.permutation import selected_block_positions
from ..utils.watermark import inverse_arnold_transform, watermark_from_bits
from .soft_decoder import _selected_r12_vectorized, nlm_versions


def _decode_blind(channel: np.ndarray, positions: np.ndarray, period: float):
    vals = _selected_r12_vectorized(channel, positions, 2)
    phase = np.mod(vals, float(period))
    bits = (phase >= 0.5 * float(period)).astype(np.uint8)
    boundary_distance = np.minimum(
        np.minimum(phase, np.abs(phase - 0.5 * float(period))),
        float(period) - phase,
    )
    conf = np.clip(boundary_distance / (0.25 * float(period)), 0.0, 1.0)
    return bits, conf


def extract_blind_image(
    image: np.ndarray,
    key: bytes,
    cfg: ProposedConfig,
    watermark_shape: tuple[int, int] | None = None,
):
    cfg.validate()
    expected = (int(cfg.watermark_size), int(cfg.watermark_size))
    if watermark_shape is not None and tuple(map(int, watermark_shape)) != expected:
        raise ValueError(
            f"blind decoder shape is fixed by the public config as {expected}, got {watermark_shape}"
        )
    shape = expected
    n = int(shape[0] * shape[1])
    channel = np.asarray(image)[:, :, cfg.channel].astype(np.float64)
    h0 = (channel.shape[0] // 2) * 2
    w0 = (channel.shape[1] // 2) * 2
    positions = np.asarray(
        selected_block_positions((h0, w0), 2, n, key), dtype=np.int64
    )

    candidates = []
    for name, work in nlm_versions(channel[:h0, :w0], cfg):
        bits, conf = _decode_blind(work, positions, float(cfg.qim_period))
        candidates.append((float(np.mean(conf)), name, bits, conf))
    score, name, bits, conf = max(candidates, key=lambda x: x[0])
    recovered = inverse_arnold_transform(
        watermark_from_bits(bits, shape), int(cfg.arnold_iterations)
    )
    return recovered, score, {
        "method": "blind_mecqr_qim_v3",
        "algorithm_revision": "fully_blind_safe_projection_v3",
        "fully_blind": True,
        "side_information_used": False,
        "original_host_used": False,
        "selected_preprocess": name,
        "bit_confidence_mean": float(np.mean(conf)),
        "qim_period": float(cfg.qim_period),
        "blind_projection": str(cfg.blind_projection),
        "blind_margin_ratio": float(cfg.blind_margin_ratio),
        "payload_embeddings_per_bit": 1,
    }
