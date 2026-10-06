from __future__ import annotations

"""Closed-form coupled Q/R carrier used by fully blind v4.

For a selected 2x2 block ``A=[a,b]`` with canonical QR factorization ``A=QR``
(``diag(R)>=0``), v4 uses two observations of the same payload bit:

1. the direction of ``q1`` (equivalently the angle of the first image column),
2. a scale-normalized triangular carrier

       u_R = asinh(r12 / sqrt(r11^2 + r22^2 + eps)).

The second statistic is invariant to a positive common gain ``A -> alpha A``.
After a target ``u_R*`` is chosen, only ``r12`` changes and the corresponding
minimum-norm image-column update is still the original QR update

       b* = b + (r12* - r12) q1.

Thus v4 keeps the QR/r12 mathematical core while adding a Q-domain observation
and removing the raw-amplitude sensitivity of the R-domain decision.
"""

from dataclasses import dataclass

import numpy as np


_EPS = 1.0  # fixed public regularizer in squared 8-bit sample units


@dataclass(frozen=True)
class CoupledGeometry:
    a0: np.ndarray
    a1: np.ndarray
    b0: np.ndarray
    b1: np.ndarray
    r11: np.ndarray
    theta: np.ndarray
    q0: np.ndarray
    q1: np.ndarray
    r12: np.ndarray
    r22: np.ndarray
    scale: np.ndarray
    normalized_r12: np.ndarray


def _as_float_selected(channel: np.ndarray, rr: np.ndarray, cc: np.ndarray):
    x = np.asarray(channel)
    rr = np.asarray(rr, dtype=np.intp)
    cc = np.asarray(cc, dtype=np.intp)
    a0 = x[rr, cc].astype(np.float64, copy=False)
    a1 = x[rr + 1, cc].astype(np.float64, copy=False)
    b0 = x[rr, cc + 1].astype(np.float64, copy=False)
    b1 = x[rr + 1, cc + 1].astype(np.float64, copy=False)
    return a0, a1, b0, b1


def geometry_from_samples(a0, a1, b0, b1) -> CoupledGeometry:
    a0 = np.asarray(a0, dtype=np.float64)
    a1 = np.asarray(a1, dtype=np.float64)
    b0 = np.asarray(b0, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    r11 = np.hypot(a0, a1)
    good = r11 > 1e-12
    den = np.where(good, r11, 1.0)
    q0 = np.where(good, a0 / den, 1.0)
    q1 = np.where(good, a1 / den, 0.0)
    theta = np.arctan2(q1, q0)
    r12 = q0 * b0 + q1 * b1
    det = a0 * b1 - a1 * b0
    r22 = np.where(good, np.abs(det) / den, np.abs(b1))
    scale = np.sqrt(r11 * r11 + r22 * r22 + _EPS)
    normalized_r12 = np.arcsinh(r12 / scale)
    return CoupledGeometry(
        a0=a0,
        a1=a1,
        b0=b0,
        b1=b1,
        r11=r11,
        theta=theta,
        q0=q0,
        q1=q1,
        r12=r12,
        r22=r22,
        scale=scale,
        normalized_r12=normalized_r12,
    )


def selected_geometry(channel: np.ndarray, rr: np.ndarray, cc: np.ndarray) -> CoupledGeometry:
    return geometry_from_samples(*_as_float_selected(channel, rr, cc))


def _safe_half_width(period: float, margin_ratio: float) -> tuple[float, float]:
    p = float(period)
    m = float(margin_ratio)
    if p <= 0.0:
        raise ValueError("QIM period must be positive")
    if not (0.0 < m < 0.25):
        raise ValueError("QIM margin ratio must lie strictly in (0,0.25)")
    rho = m * p
    return rho, 0.25 * p - rho


def center_or_safe_projection(
    values: np.ndarray,
    bits: np.ndarray,
    period: float,
    margin_ratio: float,
    feasible_lo: np.ndarray,
    feasible_hi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Choose the nearest same-bit center; fall back to its safe interval.

    The center is used whenever it is pixel-feasible because it maximizes the
    hard-decision margin.  If clipping makes the center infeasible, the value is
    projected onto the nearest intersection of a same-bit safe interval and the
    feasible feature range.  No host-dependent map is stored.
    """
    x = np.asarray(values, dtype=np.float64).ravel()
    b = np.asarray(bits, dtype=np.uint8).ravel()
    lo = np.asarray(feasible_lo, dtype=np.float64).ravel()
    hi = np.asarray(feasible_hi, dtype=np.float64).ravel()
    if not (x.size == b.size == lo.size == hi.size):
        raise ValueError("projection vectors must have matching lengths")

    p = float(period)
    rho, half_width = _safe_half_width(p, margin_ratio)
    center0 = (0.25 + 0.50 * b.astype(np.float64)) * p
    k_near = np.rint((x - center0) / p)
    center = center0 + k_near * p
    center_ok = (center >= lo) & (center <= hi)

    k_min = np.ceil((lo - center0 - half_width) / p - 1e-12)
    k_max = np.floor((hi - center0 + half_width) / p + 1e-12)
    safe_ok = k_min <= k_max
    k = np.minimum(np.maximum(k_near, k_min), k_max)
    il = np.maximum(center0 + k * p - half_width, lo)
    ih = np.minimum(center0 + k * p + half_width, hi)
    safe_target = np.minimum(np.maximum(x, il), ih)

    target = np.where(center_ok, center, safe_target)
    ok = center_ok | safe_ok
    target = np.where(ok, target, x)
    return target, ok, rho


def qim_evidence(values: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return hard bits, signed triangular evidence, and [0,1] confidence."""
    p = float(period)
    x = np.asarray(values, dtype=np.float64)
    phase = np.mod(x, p)
    half = 0.5 * p
    quarter = 0.25 * p
    bits = (phase >= half).astype(np.uint8)
    distance = np.minimum(np.minimum(phase, np.abs(phase - half)), p - phase)
    confidence = np.clip(distance / quarter, 0.0, 1.0)
    eps = np.finfo(np.float64).eps
    signed = np.where(bits != 0, np.maximum(confidence, eps), -np.maximum(confidence, eps))
    return bits, signed.astype(np.float64, copy=False), confidence.astype(np.float64, copy=False)


def second_column_delta_bounds(q0, q1, b0, b1):
    """Exact r12-displacement interval keeping both modified samples in uint8 range."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    b0 = np.asarray(b0, dtype=np.float64)
    b1 = np.asarray(b1, dtype=np.float64)
    lo = np.full(q0.shape, -np.inf, dtype=np.float64)
    hi = np.full(q0.shape, np.inf, dtype=np.float64)
    for q, x in ((q0, b0), (q1, b1)):
        nz = np.abs(q) > 1e-15
        den = np.where(nz, q, 1.0)
        z0 = (0.0 - x) / den
        z1 = (255.0 - x) / den
        lo = np.maximum(lo, np.where(nz, np.minimum(z0, z1), -np.inf))
        hi = np.minimum(hi, np.where(nz, np.maximum(z0, z1), np.inf))
    return lo, hi



def _close_q_integer_pair(x0, x1, bits, out0, out1, period: float, radius: int = 3):
    """Bounded exact uint8 closure for the Q-angle carrier.

    The continuous solution is unchanged.  Only samples whose nearest-integer
    pair crosses a QIM decision boundary are reconsidered, over a fixed local
    integer stencil.  This is a deterministic quantization closure, not an
    optimization loop, and it serializes no host-dependent state.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    bits = np.asarray(bits, dtype=np.uint8)
    o0 = np.asarray(out0, dtype=np.uint8).copy()
    o1 = np.asarray(out1, dtype=np.uint8).copy()
    theta = np.arctan2(o1.astype(np.float64), o0.astype(np.float64))
    hard, _, _ = qim_evidence(theta, period)
    bad = np.flatnonzero(hard != bits)
    if bad.size == 0:
        return o0, o1, 0

    offsets = np.arange(-int(radius), int(radius) + 1, dtype=np.int16)
    du, dv = np.meshgrid(offsets, offsets, indexing="ij")
    du = du.ravel()[None, :]
    dv = dv.ravel()[None, :]
    base0 = np.rint(x0[bad]).astype(np.int16)[:, None]
    base1 = np.rint(x1[bad]).astype(np.int16)[:, None]
    c0 = np.clip(base0 + du, 0, 255).astype(np.float64)
    c1 = np.clip(base1 + dv, 0, 255).astype(np.float64)
    cand_theta = np.arctan2(c1, c0)
    cand_bits, _, _ = qim_evidence(cand_theta, period)
    feasible = cand_bits == bits[bad, None]
    cost = (c0 - x0[bad, None]) ** 2 + (c1 - x1[bad, None]) ** 2
    cost = np.where(feasible, cost, np.inf)
    choice = np.argmin(cost, axis=1)
    ok = np.isfinite(cost[np.arange(bad.size), choice])
    if np.any(ok):
        ii = bad[ok]
        jj = choice[ok]
        o0[ii] = c0[ok, jj].astype(np.uint8)
        o1[ii] = c1[ok, jj].astype(np.uint8)
    return o0, o1, int(np.count_nonzero(ok))


def _close_r_integer_pair(a0, a1, x0, x1, bits, out0, out1, period: float, radius: int = 3):
    """Bounded exact uint8 closure for the normalized R12 carrier."""
    a0 = np.asarray(a0, dtype=np.uint8)
    a1 = np.asarray(a1, dtype=np.uint8)
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    bits = np.asarray(bits, dtype=np.uint8)
    o0 = np.asarray(out0, dtype=np.uint8).copy()
    o1 = np.asarray(out1, dtype=np.uint8).copy()
    g = geometry_from_samples(a0, a1, o0, o1)
    hard, _, _ = qim_evidence(g.normalized_r12, period)
    bad = np.flatnonzero(hard != bits)
    if bad.size == 0:
        return o0, o1, 0

    offsets = np.arange(-int(radius), int(radius) + 1, dtype=np.int16)
    du, dv = np.meshgrid(offsets, offsets, indexing="ij")
    du = du.ravel()[None, :]
    dv = dv.ravel()[None, :]
    base0 = np.rint(x0[bad]).astype(np.int16)[:, None]
    base1 = np.rint(x1[bad]).astype(np.int16)[:, None]
    c0 = np.clip(base0 + du, 0, 255).astype(np.float64)
    c1 = np.clip(base1 + dv, 0, 255).astype(np.float64)

    aa0 = a0[bad].astype(np.float64)[:, None]
    aa1 = a1[bad].astype(np.float64)[:, None]
    r11 = np.hypot(aa0, aa1)
    den = np.where(r11 > 1e-12, r11, 1.0)
    q0 = np.where(r11 > 1e-12, aa0 / den, 1.0)
    q1 = np.where(r11 > 1e-12, aa1 / den, 0.0)
    r12 = q0 * c0 + q1 * c1
    det = aa0 * c1 - aa1 * c0
    r22 = np.where(r11 > 1e-12, np.abs(det) / den, np.abs(c1))
    scale = np.sqrt(r11 * r11 + r22 * r22 + _EPS)
    u = np.arcsinh(r12 / scale)
    cand_bits, _, _ = qim_evidence(u, period)
    feasible = cand_bits == bits[bad, None]
    cost = (c0 - x0[bad, None]) ** 2 + (c1 - x1[bad, None]) ** 2
    cost = np.where(feasible, cost, np.inf)
    choice = np.argmin(cost, axis=1)
    ok = np.isfinite(cost[np.arange(bad.size), choice])
    if np.any(ok):
        ii = bad[ok]
        jj = choice[ok]
        o0[ii] = c0[ok, jj].astype(np.uint8)
        o1[ii] = c1[ok, jj].astype(np.uint8)
    return o0, o1, int(np.count_nonzero(ok))

def project_coupled_selected_pixels(
    channel: np.ndarray,
    rr: np.ndarray,
    cc: np.ndarray,
    bits: np.ndarray,
    *,
    q_period: float,
    r_period: float,
    q_margin_ratio: float,
    r_margin_ratio: float,
):
    """Embed one bit in Q direction and normalized R12 of each selected block.

    Exactly one keyed 2x2 block is used per payload bit.  Both observations live
    in that same block, so the logical 4096-bit payload is not repeated across
    additional blocks.
    """
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    rr = np.asarray(rr, dtype=np.intp)
    cc = np.asarray(cc, dtype=np.intp)
    if not (bits.size == rr.size == cc.size):
        raise ValueError("bits/carrier count mismatch")

    g0 = selected_geometry(channel, rr, cc)

    # Q carrier.  The first column is rotated at fixed norm whenever possible.
    # If the target angle would make an 8-bit sample exceed 255, shrink only the
    # radius by the minimum amount needed; the angle (and therefore the bit)
    # remains exact in the continuous solution.
    theta_lo = np.zeros(bits.size, dtype=np.float64)
    theta_hi = np.full(bits.size, 0.5 * np.pi, dtype=np.float64)
    theta_target, q_ok, q_margin = center_or_safe_projection(
        g0.theta, bits, q_period, q_margin_ratio, theta_lo, theta_hi
    )
    c = np.cos(theta_target)
    s = np.sin(theta_target)
    max_comp = np.maximum(np.abs(c), np.abs(s))
    max_radius = 255.0 / np.maximum(max_comp, 1e-12)
    radius = np.minimum(g0.r11, max_radius)
    qa0_cont = radius * c
    qa1_cont = radius * s
    qa0 = np.clip(np.rint(qa0_cont), 0, 255).astype(np.uint8)
    qa1 = np.clip(np.rint(qa1_cont), 0, 255).astype(np.uint8)
    qa0, qa1, q_closed = _close_q_integer_pair(
        qa0_cont, qa1_cont, bits, qa0, qa1, q_period
    )

    # Recompute QR geometry from the exact integer Q column before embedding R.
    # ensures the R target is defined relative to the exact samples that will be
    # present in the watermarked image.
    g1 = geometry_from_samples(qa0, qa1, g0.b0, g0.b1)
    dlo, dhi = second_column_delta_bounds(g1.q0, g1.q1, g1.b0, g1.b1)
    u_lo = np.arcsinh((g1.r12 + dlo) / g1.scale)
    u_hi = np.arcsinh((g1.r12 + dhi) / g1.scale)
    u_target, r_ok, r_margin = center_or_safe_projection(
        g1.normalized_r12, bits, r_period, r_margin_ratio, u_lo, u_hi
    )
    target_r12 = g1.scale * np.sinh(u_target)
    delta = target_r12 - g1.r12
    qb0_cont = g1.b0 + delta * g1.q0
    qb1_cont = g1.b1 + delta * g1.q1
    qb0 = np.clip(np.rint(qb0_cont), 0, 255).astype(np.uint8)
    qb1 = np.clip(np.rint(qb1_cont), 0, 255).astype(np.uint8)
    qb0, qb1, r_closed = _close_r_integer_pair(
        qa0, qa1, qb0_cont, qb1_cont, bits, qb0, qb1, r_period
    )

    # Final integer-domain observations.  The embedding uses the exact rounded
    # codeword; these counts are useful diagnostics but are never serialized.
    gf = geometry_from_samples(qa0, qa1, qb0, qb1)
    q_bits, _, q_conf = qim_evidence(gf.theta, q_period)
    r_bits, _, r_conf = qim_evidence(gf.normalized_r12, r_period)

    d_a0 = qa0.astype(np.float64) - g0.a0
    d_a1 = qa1.astype(np.float64) - g0.a1
    d_b0 = qb0.astype(np.float64) - g0.b0
    d_b1 = qb1.astype(np.float64) - g0.b1
    sse = float(np.sum(d_a0 * d_a0 + d_a1 * d_a1 + d_b0 * d_b0 + d_b1 * d_b1))
    changed = int(
        np.count_nonzero(qa0 != g0.a0)
        + np.count_nonzero(qa1 != g0.a1)
        + np.count_nonzero(qb0 != g0.b0)
        + np.count_nonzero(qb1 != g0.b1)
    )

    return {
        "a0": qa0,
        "a1": qa1,
        "b0": qb0,
        "b1": qb1,
        "sse": sse,
        "changed_samples": changed,
        "q_projection_feasible": q_ok,
        "r_projection_feasible": r_ok,
        "q_clean_match": q_bits == bits,
        "r_clean_match": r_bits == bits,
        "q_confidence": q_conf,
        "r_confidence": r_conf,
        "q_nominal_margin": float(q_margin),
        "r_nominal_margin": float(r_margin),
        "q_integer_closure_count": int(q_closed),
        "r_integer_closure_count": int(r_closed),
    }


def decode_coupled_selected_pixels(
    channel: np.ndarray,
    rr: np.ndarray,
    cc: np.ndarray,
    *,
    q_period: float,
    r_period: float,
    q_weight: float,
    r_weight: float,
    energy_tau: float,
):
    g = selected_geometry(channel, rr, cc)
    q_bits, q_score, q_conf = qim_evidence(g.theta, q_period)
    r_bits, r_score, r_conf = qim_evidence(g.normalized_r12, r_period)

    tau = max(float(energy_tau), 0.0)
    if tau == 0.0:
        q_rel = np.ones_like(g.r11)
        r_rel = np.ones_like(g.scale)
    else:
        q_rel = g.r11 / (g.r11 + tau)
        r_rel = g.scale / (g.scale + tau)

    q_term = float(q_weight) * q_score * q_rel
    r_term = float(r_weight) * r_score * r_rel
    fused = q_term + r_term
    bits = (fused >= 0.0).astype(np.uint8)
    denom = max(abs(float(q_weight)) + abs(float(r_weight)), 1e-12)
    confidence = np.clip(np.abs(fused) / denom, 0.0, 1.0)
    return bits, confidence, {
        "q_bits": q_bits,
        "r_bits": r_bits,
        "q_confidence": q_conf,
        "r_confidence": r_conf,
        "q_energy_reliability": q_rel,
        "r_energy_reliability": r_rel,
        "fused_score": fused,
    }
