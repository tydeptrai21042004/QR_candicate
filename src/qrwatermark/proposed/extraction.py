from __future__ import annotations

import numpy as np

from ..core.config import ProposedConfig
from ..utils.permutation import selected_block_positions
from ..utils.watermark import inverse_arnold_transform, watermark_from_bits
from .side_info import unpack_modes
from .soft_decoder import decode_groups, nlm_versions


def extract_image(image: np.ndarray, key: bytes, side_info: dict, cfg: ProposedConfig):
    modes, meta = unpack_modes(side_info, key)
    shape = tuple(int(x) for x in meta["watermark_shape"])
    repetition = int(meta["repetition"])
    block_size = int(meta["block_size"])
    channel_index = int(meta["channel"])
    arnold_iterations = int(meta["arnold_iterations"])
    if repetition != cfg.repetition or block_size != cfg.block_size:
        raise ValueError("Method configuration does not match authenticated side information")

    channel = np.asarray(image, dtype=np.uint8)[:, :, channel_index]
    h0 = (channel.shape[0] // block_size) * block_size
    w0 = (channel.shape[1] // block_size) * block_size
    positions = selected_block_positions((h0, w0), block_size, len(modes) * repetition, key)

    candidates = []
    for name, work in nlm_versions(channel[:h0, :w0], cfg):
        bits, conf, bit_conf = decode_groups(work, positions, modes, repetition, block_size, cfg)
        candidates.append((conf, name, bits, bit_conf))
    conf, name, bits, bit_conf = max(candidates, key=lambda x: x[0])
    scrambled = watermark_from_bits(bits, shape)
    recovered = inverse_arnold_transform(scrambled, arnold_iterations)
    return recovered, float(conf), {
        "selected_preprocess": name,
        "candidate_confidence": {n: float(c) for c, n, _, _ in candidates},
        "bit_confidence_mean": float(np.mean(bit_conf)),
    }
