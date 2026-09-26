from __future__ import annotations

import numpy as np

from .qr import canonical_qr


def _nearest_lattice(value: float, period: float, offset: float) -> float:
    k = round((value - offset) / period)
    return offset + k * period


def build_r_candidate(block: np.ndarray, bit: int, period: float) -> np.ndarray:
    """Embed in r12, avoiding canonical-sign changes on diagonal coefficients."""
    q, r = canonical_qr(block)
    r_new = r.copy()
    value = float(r_new[0, 1])
    offset = (0.25 if int(bit) == 0 else 0.75) * period
    r_new[0, 1] = _nearest_lattice(value, period, offset)
    return q @ r_new


def r_llr(block: np.ndarray, period: float) -> float:
    _, r = canonical_qr(block)
    phase = float(r[0, 1]) % period
    return float(np.sin(2.0 * np.pi * (phase - 0.5 * period) / period))
