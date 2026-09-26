from __future__ import annotations

import numpy as np

from ..core.config import ProposedConfig
from ..utils.permutation import selected_block_positions
from ..utils.watermark import inverse_arnold_transform, watermark_from_bits
from .side_info import unpack_period_indices
from .soft_decoder import decode_groups, nlm_versions


def extract_image(image: np.ndarray, key: bytes, side_info: dict, cfg: ProposedConfig):
    period_indices, meta = unpack_period_indices(side_info, key)
    shape = tuple(int(x) for x in meta["watermark_shape"])
    repetition = int(meta["repetition"])
    block_size = int(meta["block_size"])
    channel_index = int(meta["channel"])
    arnold_iterations = int(meta["arnold_iterations"])
    period_table = np.asarray(meta["period_candidates"], dtype=np.float64)

    if repetition != cfg.repetition or block_size != cfg.block_size:
        raise ValueError("Method configuration does not match authenticated side information")
    if tuple(float(x) for x in period_table) != tuple(float(x) for x in cfg.period_candidates):
        raise ValueError("QIM period table does not match authenticated side information")

    periods = period_table[period_indices]
    channel = np.asarray(image, dtype=np.uint8)[:, :, channel_index]
    h0 = (channel.shape[0] // block_size) * block_size
    w0 = (channel.shape[1] // block_size) * block_size
    positions = selected_block_positions((h0, w0), block_size, len(period_indices) * repetition, key)

    candidates = []
    for name, work in nlm_versions(channel[:h0, :w0], cfg):
        bits, conf, bit_conf = decode_groups(work, positions, periods, repetition, block_size)
        candidates.append((conf, name, bits, bit_conf))

    conf, name, bits, bit_conf = max(candidates, key=lambda x: x[0])
    scrambled = watermark_from_bits(bits, shape)
    recovered = inverse_arnold_transform(scrambled, arnold_iterations)
    return recovered, float(conf), {
        "method": "ccqr_r12_qim_v1",
        "selected_preprocess": name,
        "candidate_confidence": {n: float(c) for c, n, _, _ in candidates},
        "bit_confidence_mean": float(np.mean(bit_conf)),
        "period_histogram": {str(float(p)): int(np.sum(periods == p)) for p in np.unique(periods)},
    }
