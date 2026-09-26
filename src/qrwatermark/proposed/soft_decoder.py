from __future__ import annotations

import cv2
import numpy as np

from ..core.config import ProposedConfig
from .r_branch import r_llr


def decode_groups(
    channel: np.ndarray,
    positions: list[tuple[int, int]],
    periods: np.ndarray,
    repetition: int,
    block_size: int,
) -> tuple[np.ndarray, float, np.ndarray]:
    periods = np.asarray(periods, dtype=np.float64).ravel()
    l = periods.size
    expected = l * repetition
    if len(positions) != expected:
        raise ValueError(f"positions={len(positions)} but expected={expected}")
    sums = np.zeros(l, dtype=np.float64)
    abs_sums = np.zeros(l, dtype=np.float64)
    for k, period in enumerate(periods):
        for j in range(repetition):
            row, col = positions[k * repetition + j]
            block = channel[row : row + block_size, col : col + block_size].astype(np.float64)
            llr = r_llr(block, float(period))
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
