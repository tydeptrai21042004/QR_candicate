from __future__ import annotations

import numpy as np

from ..core.config import ProposedConfig
from ..utils.permutation import selected_block_positions
from ..utils.watermark import inverse_arnold_transform, watermark_from_bits
from .side_info import unpack_certified_mask, unpack_selector_indices
from .soft_decoder import _selected_r12_vectorized, nlm_versions


def _decode_selected(channel, chosen_positions, period):
    vals = _selected_r12_vectorized(channel, chosen_positions, 2)
    phase = np.mod(vals, float(period))
    bits = (phase >= 0.5 * float(period)).astype(np.uint8)
    quarter = 0.25 * float(period)
    conf = np.minimum(np.minimum(phase, np.abs(phase - 0.5 * period)), period - phase) / quarter
    return bits, np.clip(conf, 0.0, 1.0)


def extract_elegant_image(image: np.ndarray, key: bytes, side_info: dict, cfg: ProposedConfig):
    selectors, meta = unpack_selector_indices(side_info, key)
    shape = tuple(int(x) for x in meta["watermark_shape"])
    pool = int(meta["carrier_pool"])
    block_size = int(meta["block_size"])
    channel_index = int(meta["channel"])
    arnold_iterations = int(meta["arnold_iterations"])
    period = float(meta["qim_period"])
    if block_size != 2:
        raise ValueError("single-carrier method is derived for 2x2 blocks")
    if pool != int(cfg.carrier_pool) or abs(period - float(cfg.qim_period)) > 1e-12:
        raise ValueError("method configuration does not match authenticated side information")

    channel = np.asarray(image)[:, :, channel_index].astype(np.float64)
    h0 = (channel.shape[0] // 2) * 2
    w0 = (channel.shape[1] // 2) * 2
    positions = np.asarray(
        selected_block_positions((h0, w0), 2, selectors.size * pool, key), dtype=np.int64
    ).reshape(selectors.size, pool, 2)
    chosen = positions[np.arange(selectors.size), selectors.astype(np.int64)]

    candidates = []
    for name, work in nlm_versions(channel[:h0, :w0], cfg):
        bits, conf = _decode_selected(work, chosen, period)
        candidates.append((float(np.mean(conf)), name, bits, conf))
    score, name, bits, conf = max(candidates, key=lambda x: x[0])
    recovered = inverse_arnold_transform(watermark_from_bits(bits, shape), arnold_iterations)

    try:
        mask = unpack_certified_mask(side_info, key)
        cert_fraction = float(mask.mean()) if mask.size else 0.0
    except Exception:
        mask = np.zeros(bits.size, dtype=bool)
        cert_fraction = 0.0
    return recovered, score, {
        "method": "mecqr_qim_v2",
        "algorithm_revision": meta.get("algorithm_revision", "minimum_energy_single_carrier_v2"),
        "selected_preprocess": name,
        "bit_confidence_mean": float(np.mean(conf)),
        "qim_period": period,
        "carrier_pool": pool,
        "payload_embeddings_per_bit": 1,
        "certified_fraction": cert_fraction,
    }
