from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..core.config import ProposedConfig
from .q_branch import build_q_candidate, q_llr
from .r_branch import build_r_candidate, r_llr

MODE_Q = 0
MODE_R = 1


@dataclass(frozen=True)
class CandidateEvaluation:
    score: float
    robustness: float
    distortion: float
    raw_margin: float
    mean_margin: float
    worst_margin: float


def _margin(block: np.ndarray, mode: int, bit: int, cfg: ProposedConfig) -> float:
    llr = q_llr(block, cfg.q_angle_period) if mode == MODE_Q else r_llr(block, cfg.r12_period)
    return llr if int(bit) == 1 else -llr


def _candidate(block: np.ndarray, mode: int, bit: int, cfg: ProposedConfig) -> np.ndarray:
    return (
        build_q_candidate(block, bit, cfg.q_angle_period)
        if mode == MODE_Q
        else build_r_candidate(block, bit, cfg.r12_period)
    )


def evaluate_candidate(block: np.ndarray, mode: int, bit: int, cfg: ProposedConfig) -> CandidateEvaluation:
    cand = _candidate(block, mode, bit, cfg)
    distortion = float(np.mean((np.asarray(block, dtype=np.float64) - cand) ** 2))
    base = np.clip(np.rint(cand), 0, 255).astype(np.float64)
    d = float(cfg.perturb_plus_minus)
    variants = [base, np.clip(base + d, 0, 255), np.clip(base - d, 0, 255)]
    blur = cv2.GaussianBlur(
        base.astype(np.float32),
        (3, 3),
        float(cfg.perturb_blur_sigma),
        borderType=cv2.BORDER_REFLECT_101,
    )
    variants.append(np.clip(np.rint(blur), 0, 255).astype(np.float64))
    margins = np.asarray([_margin(v, mode, bit, cfg) for v in variants], dtype=np.float64)
    raw = float(margins[0])
    mean = float(margins.mean())
    worst = float(margins.min())
    robustness = 0.5 * raw + 0.3 * mean + 0.2 * worst
    d_norm = distortion / (distortion + float(cfg.distortion_scale))
    score = robustness - float(cfg.lambda_distortion) * d_norm
    return CandidateEvaluation(score, robustness, distortion, raw, mean, worst)


def bit_neutral_utility(block: np.ndarray, mode: int, cfg: ProposedConfig) -> float:
    e0 = evaluate_candidate(block, mode, 0, cfg)
    e1 = evaluate_candidate(block, mode, 1, cfg)
    mean = 0.5 * (e0.score + e1.score)
    imbalance = abs(e0.score - e1.score)
    # The actual payload bit is deliberately absent from this function.
    return float(mean - cfg.utility_imbalance_penalty * imbalance)


def choose_mode_for_group(blocks: list[np.ndarray], cfg: ProposedConfig) -> tuple[int, dict[str, float]]:
    uq = np.asarray([bit_neutral_utility(b, MODE_Q, cfg) for b in blocks], dtype=np.float64)
    ur = np.asarray([bit_neutral_utility(b, MODE_R, cfg) for b in blocks], dtype=np.float64)
    q_score = float(uq.mean())
    r_score = float(ur.mean())
    mode = MODE_Q if q_score >= r_score else MODE_R
    return mode, {"q_utility": q_score, "r_utility": r_score}


def build_candidate(block: np.ndarray, mode: int, bit: int, cfg: ProposedConfig) -> np.ndarray:
    return _candidate(block, mode, bit, cfg)
