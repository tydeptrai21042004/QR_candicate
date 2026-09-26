from __future__ import annotations

import hashlib

import numpy as np

from ..core.config import ProposedConfig
from ..utils.permutation import selected_block_positions
from ..utils.watermark import arnold_transform, bits_from_watermark
from .branch_selector import MODE_Q, MODE_R, build_candidate, choose_mode_for_group
from .side_info import build_side_info


def embed_image(host: np.ndarray, watermark: np.ndarray, key: bytes, cfg: ProposedConfig):
    cfg.validate()
    host = np.asarray(host, dtype=np.uint8)
    if host.ndim != 3 or host.shape[2] < 3:
        raise ValueError("host must be a color image")
    wm = np.asarray(watermark, dtype=np.uint8)
    if wm.shape != (cfg.watermark_size, cfg.watermark_size):
        raise ValueError(f"watermark must be {cfg.watermark_size}x{cfg.watermark_size}")

    scrambled = arnold_transform(wm, cfg.arnold_iterations)
    bits = bits_from_watermark(scrambled)
    l = bits.size
    channel = host[:, :, cfg.channel].astype(np.float64).copy()
    h0 = (channel.shape[0] // cfg.block_size) * cfg.block_size
    w0 = (channel.shape[1] // cfg.block_size) * cfg.block_size
    count = l * cfg.repetition
    positions = selected_block_positions((h0, w0), cfg.block_size, count, key)

    # First pass: choose one Q/R mode per payload bit without seeing that bit.
    modes = np.zeros(l, dtype=np.uint8)
    utilities: list[dict[str, float]] = []
    for k in range(l):
        blocks = []
        for j in range(cfg.repetition):
            row, col = positions[k * cfg.repetition + j]
            blocks.append(channel[row : row + cfg.block_size, col : col + cfg.block_size].copy())
        mode, util = choose_mode_for_group(blocks, cfg)
        modes[k] = mode
        utilities.append(util)

    # Second pass: only now is the actual watermark bit used.
    for k, bit in enumerate(bits):
        mode = int(modes[k])
        for j in range(cfg.repetition):
            row, col = positions[k * cfg.repetition + j]
            block = channel[row : row + cfg.block_size, col : col + cfg.block_size].copy()
            cand = build_candidate(block, mode, int(bit), cfg)
            channel[row : row + cfg.block_size, col : col + cfg.block_size] = np.clip(np.rint(cand), 0, 255)

    out = host.copy()
    out[:h0, :w0, cfg.channel] = channel[:h0, :w0].astype(np.uint8)
    metadata = {
        "watermark_shape": [cfg.watermark_size, cfg.watermark_size],
        "repetition": cfg.repetition,
        "block_size": cfg.block_size,
        "channel": cfg.channel,
        "arnold_iterations": cfg.arnold_iterations,
        "key_fingerprint": hashlib.sha256(key).hexdigest()[:16],
    }
    side = build_side_info(modes, metadata, key)
    q_count = int(np.sum(modes == MODE_Q))
    r_count = int(np.sum(modes == MODE_R))
    return out, side, {
        "q_modes": q_count,
        "r_modes": r_count,
        "q_ratio": q_count / float(l),
        "r_ratio": r_count / float(l),
        "mean_q_utility": float(np.mean([u["q_utility"] for u in utilities])),
        "mean_r_utility": float(np.mean([u["r_utility"] for u in utilities])),
    }
