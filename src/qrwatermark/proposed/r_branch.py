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


def apply_r12_delta(block: np.ndarray, delta: float) -> np.ndarray:
    """Apply an exact canonical-QR ``r12`` displacement without a QR solve.

    For a 2x2 block ``A=[a,b]`` with ``||a||>0``, canonical QR gives
    ``q1=a/||a||`` and changing only ``r12`` by ``delta`` changes the second
    image column by ``delta*q1``.  This is algebraically identical to
    reconstructing ``Q @ R_new`` but maps directly to a small hardware datapath
    (dot/norm/reciprocal-sqrt plus two multiply-adds).

    The degenerate zero-first-column case retains the old QR fallback so the
    public behavior stays defined for every input.
    """
    a = np.asarray(block, dtype=np.float64)
    if a.shape != (2, 2):
        raise ValueError("apply_r12_delta is implemented for 2x2 blocks")
    first = a[:, 0]
    norm = float(np.linalg.norm(first))
    if norm > 1e-12:
        out = a.copy()
        out[:, 1] = out[:, 1] + float(delta) * (first / norm)
        # NumPy QR and the algebraic formula can differ by a few ulps.  That is
        # irrelevant analytically, but at an exact k+0.5 pixel value it can
        # change bankers-rounding by one code value.  Preserve legacy software
        # bit-exactness only for this vanishingly rare boundary case; the normal
        # (and hardware) path remains the closed-form update above.
        frac = out[:, 1] - np.floor(out[:, 1])
        if np.any(np.abs(frac - 0.5) <= 1e-10):
            q, r = canonical_qr(a)
            rr = r.copy()
            rr[0, 1] += float(delta)
            return q @ rr
        return out

    # Rare degenerate fallback: preserve the previous canonical-QR semantics.
    q, r = canonical_qr(a)
    rr = r.copy()
    rr[0, 1] += float(delta)
    return q @ rr


def qim_phase_decision(statistic: float, period: float) -> tuple[int, float, float]:
    """Hardware-friendly binary-QIM decision and triangular confidence.

    Returns ``(bit, signed_score, confidence)``.  The hard decision is exactly
    the two-coset QIM decision: phases in ``[0, Delta/2)`` decode to zero and
    phases in ``[Delta/2, Delta)`` decode to one.  Confidence is the normalized
    distance to the nearest decision boundary, so no trigonometric unit is
    required in an implementation.
    """
    p = float(period)
    if p <= 0:
        raise ValueError("period must be positive")
    phase = float(statistic) % p
    half = 0.5 * p
    quarter = 0.25 * p
    bit = int(phase >= half)
    distance = min(phase, abs(phase - half), p - phase)
    confidence = float(min(1.0, max(0.0, distance / quarter)))
    # Keep an unambiguous sign even exactly on a decision boundary.
    eps = float(np.finfo(np.float64).eps)
    signed = max(confidence, eps) if bit else -max(confidence, eps)
    return bit, float(signed), confidence


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


def r12_displacement_interval(block: np.ndarray, pixel_min: float = 0.0, pixel_max: float = 255.0) -> tuple[float, float]:
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
    dlo, dhi = r12_displacement_interval(block, pixel_min, pixel_max)
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
    value = r12_value(block)
    target, _ = qim_target_for_block(block, int(bit), float(period))
    return apply_r12_delta(block, float(target - value))


def r_llr(block: np.ndarray, period: float) -> float:
    """Signed triangular QIM evidence: negative favours 0, positive favours 1."""
    _, score, _ = qim_phase_decision(r12_value(block), period)
    return float(score)
