from __future__ import annotations

"""FPGA-oriented 2x2 blind-v3 datapath helpers.

These routines keep the exact mathematical carrier used by blind-v3 while
expressing it as four scalar pixels per selected block.  The default floating
path is algorithm-equivalent to the original NumPy QR geometry.  A Q18 model
is also provided as a bit-accurate deployment reference for FPGA validation.

The online hardware datapath needs only

    a0 = x[r,c],     b0 = x[r,c+1]
    a1 = x[r+1,c],   b1 = x[r+1,c+1]

where the first column ``a`` is unchanged and only ``b`` can be modified.
"""

import math
import numpy as np


DEFAULT_FPGA_FRAC_BITS = 18


def gather_selected_pixels(channel: np.ndarray, rr: np.ndarray, cc: np.ndarray):
    """Gather selected 2x2 blocks as four vectors, converting only those pixels."""
    x = np.asarray(channel)
    rr = np.asarray(rr, dtype=np.intp)
    cc = np.asarray(cc, dtype=np.intp)
    if rr.shape != cc.shape:
        raise ValueError("row/column vectors must have matching shapes")
    a0 = x[rr, cc].astype(np.float64, copy=False)
    a1 = x[rr + 1, cc].astype(np.float64, copy=False)
    b0 = x[rr, cc + 1].astype(np.float64, copy=False)
    b1 = x[rr + 1, cc + 1].astype(np.float64, copy=False)
    return a0, a1, b0, b1


def selected_r12_float(channel: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> np.ndarray:
    """Canonical 2x2 ``r12`` using the scalar form suitable for FPGA mapping."""
    a0, a1, b0, b1 = gather_selected_pixels(channel, rr, cc)
    norm = np.sqrt(a0 * a0 + a1 * a1)
    vals = np.empty(norm.size, dtype=np.float64)
    good = norm > 1e-12
    vals[good] = (a0[good] * b0[good] + a1[good] * b1[good]) / norm[good]
    # Exact canonical convention for a zero first column: q1=[1,0].
    vals[~good] = b0[~good]
    return vals


def project_safe_selected_pixels(
    channel: np.ndarray,
    rr: np.ndarray,
    cc: np.ndarray,
    bits: np.ndarray,
    period: float,
    margin_ratio: float,
):
    """Exact floating reference for the streaming blind-v3 safe-set projector.

    It performs the same projection as the previous block-matrix path but
    returns only the two modified second-column samples.  First-column samples
    are provably unchanged and therefore never need to be written back.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if bits.size != np.asarray(rr).size:
        raise ValueError("bits/carrier count mismatch")
    period = float(period)
    ratio = float(margin_ratio)
    if period <= 0.0:
        raise ValueError("qim_period must be positive")
    if not (0.0 < ratio < 0.25):
        raise ValueError("blind_margin_ratio must lie strictly in (0, 0.25)")

    a0, a1, b0, b1 = gather_selected_pixels(channel, rr, cc)
    norm = np.sqrt(a0 * a0 + a1 * a1)
    good = norm > 1e-12

    q0 = np.empty_like(norm)
    q1 = np.empty_like(norm)
    r12 = np.empty_like(norm)
    q0[good] = a0[good] / norm[good]
    q1[good] = a1[good] / norm[good]
    r12[good] = (a0[good] * b0[good] + a1[good] * b1[good]) / norm[good]
    q0[~good] = 1.0
    q1[~good] = 0.0
    r12[~good] = b0[~good]

    lo = np.full(norm.shape, -np.inf, dtype=np.float64)
    hi = np.full(norm.shape, np.inf, dtype=np.float64)
    for q, x in ((q0, b0), (q1, b1)):
        nz = np.abs(q) > 1e-15
        z0 = np.zeros_like(q)
        z1 = np.zeros_like(q)
        z0[nz] = (0.0 - x[nz]) / q[nz]
        z1[nz] = (255.0 - x[nz]) / q[nz]
        lo = np.maximum(lo, np.where(nz, np.minimum(z0, z1), -np.inf))
        hi = np.minimum(hi, np.where(nz, np.maximum(z0, z1), np.inf))

    rho = ratio * period
    half_width = 0.25 * period - rho
    center = (0.25 + 0.50 * bits.astype(np.float64)) * period
    feasible_lo = r12 + lo
    feasible_hi = r12 + hi
    k_near = np.rint((r12 - center) / period)
    k_min = np.ceil((feasible_lo - center - half_width) / period - 1e-12)
    k_max = np.floor((feasible_hi - center + half_width) / period + 1e-12)
    feasible = k_min <= k_max
    if not np.all(feasible):
        count = int(np.sum(~feasible))
        raise ValueError(
            f"{count} keyed carriers cannot realize the requested blind safe set; "
            "reduce qim_period or blind_margin_ratio"
        )

    k = np.minimum(np.maximum(k_near, k_min), k_max)
    interval_lo = np.maximum(center + k * period - half_width, feasible_lo)
    interval_hi = np.minimum(center + k * period + half_width, feasible_hi)
    target = np.minimum(np.maximum(r12, interval_lo), interval_hi)
    delta = target - r12

    y0 = np.clip(np.rint(b0 + delta * q0), 0, 255).astype(np.uint8)
    y1 = np.clip(np.rint(b1 + delta * q1), 0, 255).astype(np.uint8)

    # Explicit uint8-decoding check, identical to the original implementation.
    ynorm = norm
    check = np.empty_like(r12)
    check[good] = (a0[good] * y0[good] + a1[good] * y1[good]) / ynorm[good]
    check[~good] = y0[~good]
    decoded = (np.mod(check, period) >= 0.5 * period).astype(np.uint8)
    if not np.all(decoded == bits):
        count = int(np.sum(decoded != bits))
        raise ValueError(
            f"{count} carriers lost their bit under uint8 rounding; increase "
            "blind_margin_ratio or qim_period"
        )

    # Exact RGB-domain SSE: only these two samples are changed.
    d0 = y0.astype(np.float64) - b0
    d1 = y1.astype(np.float64) - b1
    sse = float(np.sum(d0 * d0 + d1 * d1))
    changed = int(np.count_nonzero(y0 != b0) + np.count_nonzero(y1 != b1))
    return y0, y1, delta, rho, sse, changed


def _div_round_nearest_even(num: int, den: int) -> int:
    if den <= 0:
        raise ValueError("denominator must be positive")
    sign = -1 if num < 0 else 1
    n = abs(int(num))
    q, r = divmod(n, int(den))
    twice = 2 * r
    if twice > den or (twice == den and (q & 1)):
        q += 1
    return sign * q


def _isqrt_round_nearest_even(value: int) -> int:
    root = math.isqrt(int(value))
    lo = int(value) - root * root
    hi = (root + 1) * (root + 1) - int(value)
    if lo > hi or (lo == hi and (root & 1)):
        root += 1
    return root


def _ceil_div(num: int, den: int) -> int:
    return -((-int(num)) // int(den))


def project_safe_block_q18(
    a0: int,
    a1: int,
    b0: int,
    b1: int,
    bit: int,
    frac_bits: int = DEFAULT_FPGA_FRAC_BITS,
) -> tuple[int, int]:
    """Integer Q18 deployment reference for the default blind-v3 parameters.

    This model is intentionally specialized to the paper/online configuration
    ``Delta=48`` and ``margin_ratio=1/8``.  On the bundled 12 host/watermark
    pairs it is bit-for-bit identical to the floating reference at Q18.
    """
    f = int(frac_bits)
    if f < 1:
        raise ValueError("frac_bits must be positive")
    s = 1 << f
    period = 48 * s
    half_width = 6 * s
    center = (12 if int(bit) == 0 else 36) * s

    a0 = int(a0); a1 = int(a1); b0 = int(b0); b1 = int(b1)
    norm2 = a0 * a0 + a1 * a1
    if norm2 == 0:
        q0 = s
        q1 = 0
        r12 = b0 * s
    else:
        norm = _isqrt_round_nearest_even(norm2 * s * s)
        q0 = _div_round_nearest_even(a0 * s * s, norm)
        q1 = _div_round_nearest_even(a1 * s * s, norm)
        dot = a0 * b0 + a1 * b1
        r12 = _div_round_nearest_even(dot * s * s, norm)

    neg_inf = -(1 << 62)
    pos_inf = 1 << 62
    lo, hi = neg_inf, pos_inf
    for q, x in ((q0, b0), (q1, b1)):
        if q != 0:
            z0 = _div_round_nearest_even((-x) * s * s, q)
            z1 = _div_round_nearest_even((255 - x) * s * s, q)
            lo = max(lo, min(z0, z1))
            hi = min(hi, max(z0, z1))

    feasible_lo = r12 + lo
    feasible_hi = r12 + hi
    k_near = _div_round_nearest_even(r12 - center, period)
    k_min = _ceil_div(feasible_lo - center - half_width, period)
    k_max = (feasible_hi - center + half_width) // period
    if k_min > k_max:
        raise ValueError("Q18 safe-set projection is infeasible")
    k = min(max(k_near, k_min), k_max)
    interval_lo = max(center + k * period - half_width, feasible_lo)
    interval_hi = min(center + k * period + half_width, feasible_hi)
    target = min(max(r12, interval_lo), interval_hi)
    delta = target - r12

    p0 = _div_round_nearest_even(delta * q0, s)
    p1 = _div_round_nearest_even(delta * q1, s)
    y0 = b0 + _div_round_nearest_even(p0, s)
    y1 = b1 + _div_round_nearest_even(p1, s)
    return max(0, min(255, y0)), max(0, min(255, y1))


def selected_r12_q18_bits(
    channel: np.ndarray,
    rr: np.ndarray,
    cc: np.ndarray,
    frac_bits: int = DEFAULT_FPGA_FRAC_BITS,
) -> np.ndarray:
    """Bit decisions from the integer Q18 FPGA decoder reference.

    Specialized to ``Delta=48``.  This keeps the decoder free of floating-point
    modulo/division in the hardware-facing reference model.
    """
    x = np.asarray(channel)
    rr = np.asarray(rr, dtype=np.intp)
    cc = np.asarray(cc, dtype=np.intp)
    f = int(frac_bits)
    s = 1 << f
    period = 48 * s
    half = 24 * s
    out = np.empty(rr.size, dtype=np.uint8)
    for i, (r, c) in enumerate(zip(rr.tolist(), cc.tolist())):
        a0 = int(x[r, c]); a1 = int(x[r + 1, c])
        b0 = int(x[r, c + 1]); b1 = int(x[r + 1, c + 1])
        norm2 = a0 * a0 + a1 * a1
        if norm2 == 0:
            r12 = b0 * s
        else:
            norm = _isqrt_round_nearest_even(norm2 * s * s)
            dot = a0 * b0 + a1 * b1
            r12 = _div_round_nearest_even(dot * s * s, norm)
        out[i] = 1 if (r12 % period) >= half else 0
    return out
