from __future__ import annotations

import cv2
import numpy as np

from ..core.config import ProposedConfig
from .branch_selector import MODE_Q, MODE_R
from .q_branch import q_llr
from .r_branch import r_llr


def observation_llr(block: np.ndarray, mode: int, cfg: ProposedConfig) -> float:
    if mode == MODE_Q:
        return q_llr(block, cfg.q_angle_period)
    if mode == MODE_R:
        return r_llr(block, cfg.r12_period)
    raise ValueError(f"Unknown mode: {mode}")


def decode_groups(
    channel: np.ndarray,
    positions: list[tuple[int, int]],
    modes: np.ndarray,
    repetition: int,
    block_size: int,
    cfg: ProposedConfig,
) -> tuple[np.ndarray, float, np.ndarray]:
    l = len(modes)
    expected = l * repetition
    if len(positions) != expected:
        raise ValueError(f"positions={len(positions)} but expected={expected}")
    sums = np.zeros(l, dtype=np.float64)
    abs_sums = np.zeros(l, dtype=np.float64)
    for k in range(l):
        mode = int(modes[k])
        for j in range(repetition):
            row, col = positions[k * repetition + j]
            block = channel[row : row + block_size, col : col + block_size].astype(np.float64)
            llr = observation_llr(block, mode, cfg)
            sums[k] += llr
            abs_sums[k] += abs(llr)
    bits = (sums >= 0.0).astype(np.uint8)
    bit_conf = np.abs(sums) / (abs_sums + 1e-12)
    return bits, float(bit_conf.mean()), bit_conf


def nlm_versions(channel: np.ndarray, cfg: ProposedConfig) -> list[tuple[str, np.ndarray]]:
    raw = channel.astype(np.uint8)
    versions = [("raw", raw)]
    if not cfg.nlm.enabled:
        return versions
    mild = cv2.fastNlMeansDenoising(
        raw, None, h=float(cfg.nlm.mild_h), templateWindowSize=cfg.nlm.mild_template, searchWindowSize=cfg.nlm.mild_search
    )
    strong = cv2.fastNlMeansDenoising(
        raw, None, h=float(cfg.nlm.strong_h), templateWindowSize=cfg.nlm.strong_template, searchWindowSize=cfg.nlm.strong_search
    )
    versions.extend([("nlm_mild", mild), ("nlm_strong", strong)])
    return versions
