from __future__ import annotations

import numpy as np

from .qr import canonical_qr, q_angle, q_det_sign, q_from_angle


def _nearest_lattice(value: float, period: float, offset: float) -> float:
    k = round((value - offset) / period)
    return offset + k * period


def build_q_candidate(block: np.ndarray, bit: int, period: float) -> np.ndarray:
    q, r = canonical_qr(block)
    theta = q_angle(q)
    offset = (0.25 if int(bit) == 0 else 0.75) * period
    target = _nearest_lattice(theta, period, offset)
    q_new = q_from_angle(target, q_det_sign(q))
    return q_new @ r


def q_llr(block: np.ndarray, period: float) -> float:
    q, _ = canonical_qr(block)
    phase = q_angle(q) % period
    # -1 at bit-0 center (T/4), +1 at bit-1 center (3T/4).
    return float(np.sin(2.0 * np.pi * (phase - 0.5 * period) / period))
