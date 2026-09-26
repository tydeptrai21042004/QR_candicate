from __future__ import annotations

import math
import numpy as np

from .qr import canonical_qr


def r12_value(block: np.ndarray) -> float:
    """Canonical QR R(1,2) statistic used by the proposal."""
    a = np.asarray(block, dtype=np.float64)
    first = a[:, 0]
    norm = float(np.linalg.norm(first))
    if norm > 1e-12:
        return float(np.dot(first, a[:, 1]) / norm)
    _, r = canonical_qr(a)
    return float(r[0, 1])


def _nearest_lattice(value: float, period: float, offset: float) -> float:
    k = round((value - offset) / period)
    return offset + k * period


def qim_target(value: float, bit: int, period: float) -> float:
    if period <= 0:
        raise ValueError("period must be positive")
    offset = (0.25 if int(bit) == 0 else 0.75) * float(period)
    return _nearest_lattice(float(value), float(period), offset)


def qim_displacement(value: float, bit: int, period: float) -> float:
    return qim_target(value, bit, period) - float(value)


def _delta_interval_for_valid_pixels(block: np.ndarray, pixel_min: float, pixel_max: float) -> tuple[float, float]:
    """Exact interval of R12 displacements that keep the reconstructed 2nd column in range.

    Changing only r12 by delta changes the second image-block column by
    delta*q1, while the first column is unchanged.
    """
    a = np.asarray(block, dtype=np.float64)
    first = a[:, 0]
    norm = float(np.linalg.norm(first))
    if norm <= 1e-12:
        q, _ = canonical_qr(a)
        q1 = q[:, 0]
    else:
        q1 = first / norm
    second = a[:, 1]
    lo, hi = -np.inf, np.inf
    for qi, xi in zip(q1, second):
        if abs(float(qi)) <= 1e-15:
            if xi < pixel_min or xi > pixel_max:
                return 1.0, 0.0
            continue
        a0 = (pixel_min - float(xi)) / float(qi)
        a1 = (pixel_max - float(xi)) / float(qi)
        lo = max(lo, min(a0, a1))
        hi = min(hi, max(a0, a1))
    return float(lo), float(hi)


def qim_target_for_block(
    block: np.ndarray,
    bit: int,
    period: float,
    pixel_min: float = 0.0,
    pixel_max: float = 255.0,
) -> tuple[float, bool]:
    """Nearest same-bit QIM lattice point whose reconstructed block is range-feasible."""
    value = r12_value(block)
    offset = (0.25 if int(bit) == 0 else 0.75) * float(period)
    dlo, dhi = _delta_interval_for_valid_pixels(block, pixel_min, pixel_max)
    if dlo <= dhi:
        if not np.isfinite(dlo) and not np.isfinite(dhi):
            return qim_target(value, bit, period), True
        target_lo = value + dlo
        target_hi = value + dhi
        k_lo = -10**18 if not np.isfinite(target_lo) else math.ceil((target_lo - offset) / float(period) - 1e-12)
        k_hi = 10**18 if not np.isfinite(target_hi) else math.floor((target_hi - offset) / float(period) + 1e-12)
        if k_lo <= k_hi:
            k_near = round((value - offset) / float(period))
            k = min(max(k_near, k_lo), k_hi)
            return float(offset + k * float(period)), True
    return qim_target(value, bit, period), False


def qim_displacement_for_block(block: np.ndarray, bit: int, period: float) -> tuple[float, bool]:
    target, feasible = qim_target_for_block(block, bit, period)
    return float(target - r12_value(block)), feasible


def build_r_candidate(block: np.ndarray, bit: int, period: float) -> np.ndarray:
    """Range-aware R12-QIM embedding with canonical QR diagonal unchanged."""
    q, r = canonical_qr(block)
    r_new = r.copy()
    target, _ = qim_target_for_block(block, int(bit), float(period))
    r_new[0, 1] = target
    return q @ r_new


def r_llr(block: np.ndarray, period: float) -> float:
    """Soft signed QIM evidence: negative favours bit 0, positive bit 1."""
    phase = r12_value(block) % float(period)
    return float(np.sin(2.0 * np.pi * (phase - 0.5 * period) / period))
