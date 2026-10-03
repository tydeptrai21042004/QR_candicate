from __future__ import annotations

"""Minimum-energy single-carrier QR-QIM embedding.

For every payload bit exactly one 2x2 carrier is modified.  A keyed candidate
set is used only to *choose* the minimum-energy carrier; it is not repetition
and no payload bit is embedded more than once.
"""

import hashlib
from dataclasses import replace

import numpy as np

from ..core.config import ProposedConfig
from ..utils.permutation import selected_block_positions
from ..utils.watermark import arnold_transform, bits_from_watermark
from .certificate import certify_spread_groups_arrays
from .convolution import convolution_bank, reflect_convolve
from .qr import canonical_qr
from .side_info import build_selector_side_info, side_info_overhead_bits


def _gather_blocks(channel: np.ndarray, positions: list[tuple[int, int]]) -> np.ndarray:
    pos = np.asarray(positions, dtype=np.int64)
    rr, cc = pos[:, 0], pos[:, 1]
    x = np.asarray(channel, dtype=np.float64)
    out = np.empty((len(positions), 2, 2), dtype=np.float64)
    out[:, 0, 0] = x[rr, cc]
    out[:, 0, 1] = x[rr, cc + 1]
    out[:, 1, 0] = x[rr + 1, cc]
    out[:, 1, 1] = x[rr + 1, cc + 1]
    return out


def _candidate_geometry(blocks: np.ndarray):
    """Return canonical q1, r12 and exact feasible delta intervals in batch."""
    first = blocks[..., :, 0]
    second = blocks[..., :, 1]
    norm = np.sqrt(np.sum(first * first, axis=-1))
    q1 = np.zeros_like(first)
    good = norm > 1e-12
    q1[good] = first[good] / norm[good, None]
    r12 = np.zeros(norm.shape, dtype=np.float64)
    r12[good] = np.sum(first[good] * second[good], axis=-1) / norm[good]

    # Degenerate blocks are rare but retain exact canonical-QR semantics.
    if np.any(~good):
        for idx in np.argwhere(~good):
            ii = tuple(int(v) for v in idx)
            q, r = canonical_qr(blocks[ii])
            q1[ii] = q[:, 0]
            r12[ii] = r[0, 1]

    lo = np.full(norm.shape, -np.inf, dtype=np.float64)
    hi = np.full(norm.shape, np.inf, dtype=np.float64)
    for row in range(2):
        qi = q1[..., row]
        xi = second[..., row]
        nz = np.abs(qi) > 1e-15
        a0 = np.zeros_like(qi)
        a1 = np.zeros_like(qi)
        a0[nz] = (0.0 - xi[nz]) / qi[nz]
        a1[nz] = (255.0 - xi[nz]) / qi[nz]
        lo = np.maximum(lo, np.where(nz, np.minimum(a0, a1), -np.inf))
        hi = np.minimum(hi, np.where(nz, np.maximum(a0, a1), np.inf))
    return q1, r12, lo, hi


def _embed_candidate_bank(base: np.ndarray, bits: np.ndarray, period: float):
    """Closed-form projection of every candidate onto the requested QIM coset."""
    n, pool = base.shape[:2]
    q1, r12, lo, hi = _candidate_geometry(base)
    offsets = (0.25 + 0.50 * bits.astype(np.float64))[:, None] * float(period)
    # Orthogonal projection onto the *feasible part* of the requested QIM
    # lattice.  This is the vector form of min |delta| subject to
    # r12+delta in Lambda_bit and delta in [lo,hi].
    knear = np.rint((r12 - offsets) / float(period))
    target_lo = r12 + lo
    target_hi = r12 + hi
    klo = np.where(
        np.isfinite(target_lo),
        np.ceil((target_lo - offsets) / float(period) - 1e-12),
        -1e18,
    )
    khi = np.where(
        np.isfinite(target_hi),
        np.floor((target_hi - offsets) / float(period) + 1e-12),
        1e18,
    )
    lattice_ok = klo <= khi
    kval = np.minimum(np.maximum(knear, klo), khi)
    target = offsets + kval * float(period)
    delta = target - r12
    range_ok = lattice_ok & (delta >= lo - 1e-10) & (delta <= hi + 1e-10)

    continuous = base.copy()
    continuous[..., :, 1] += delta[..., None] * q1
    rounded = np.clip(np.rint(continuous), 0, 255).astype(np.float64)

    # The optimization includes exact uint8 decodability, not an after-the-fact
    # fallback rule.  This makes the finite optimization match the implemented
    # decoder exactly.
    _, rr12, _, _ = _candidate_geometry(rounded)
    phase = np.mod(rr12, float(period))
    decoded = (phase >= 0.5 * float(period)).astype(np.uint8)
    decode_ok = decoded == bits[:, None]
    feasible = range_ok & decode_ok

    continuous_energy = delta * delta  # ||delta*q1||_2^2 because ||q1||_2=1
    rounded_sse = np.sum((rounded - base) ** 2, axis=(2, 3))
    return continuous, rounded, continuous_energy, rounded_sse, feasible


def _select_carriers(energy: np.ndarray, rounded_sse: np.ndarray, feasible: np.ndarray, policy: str):
    n, pool = energy.shape
    huge = np.finfo(np.float64).max / 16.0
    if policy == "min_energy":
        # Primary objective is exactly the mathematical delta^2 energy;
        # rounded SSE is a deterministic tie-break only.
        score = np.where(feasible, energy + 1e-12 * rounded_sse, huge)
    elif policy == "min_rounded_sse":
        score = np.where(feasible, rounded_sse, huge)
    elif policy == "first":
        score = np.where(feasible, np.arange(pool, dtype=np.float64)[None, :], huge)
    else:
        raise ValueError(f"unknown carrier_selector_policy={policy!r}")
    selectors = np.argmin(score, axis=1).astype(np.uint8)
    ok = np.any(feasible, axis=1)
    if not np.all(ok):
        count = int(np.sum(~ok))
        raise ValueError(
            f"{count} payload bits have no feasible single carrier; "
            "reduce qim_period or increase carrier_pool"
        )
    return selectors


def _selected_from_bank(bank: np.ndarray, selectors: np.ndarray) -> np.ndarray:
    return bank[np.arange(selectors.size), selectors.astype(np.int64)]


def _gather_chosen(channel: np.ndarray, chosen_positions: np.ndarray) -> np.ndarray:
    rr, cc = chosen_positions[:, 0], chosen_positions[:, 1]
    x = np.asarray(channel, dtype=np.float64)
    out = np.empty((chosen_positions.shape[0], 1, 2, 2), dtype=np.float64)
    out[:, 0, 0, 0] = x[rr, cc]
    out[:, 0, 0, 1] = x[rr, cc + 1]
    out[:, 0, 1, 0] = x[rr + 1, cc]
    out[:, 0, 1, 1] = x[rr + 1, cc + 1]
    return out


def embed_elegant_image(host, watermark, key: bytes, cfg: ProposedConfig):
    cfg.validate()
    host = np.asarray(host, dtype=np.uint8)
    wm = np.asarray(watermark, dtype=np.uint8)
    if host.ndim != 3 or host.shape[2] < 3:
        raise ValueError("host must be color")
    if wm.shape != (cfg.watermark_size, cfg.watermark_size):
        raise ValueError(f"watermark must be {cfg.watermark_size}x{cfg.watermark_size}")

    bits = bits_from_watermark(arnold_transform(wm, cfg.arnold_iterations)).astype(np.uint8)
    n = int(bits.size)
    pool = int(cfg.carrier_pool)
    period = float(cfg.qim_period)
    channel = host[:, :, cfg.channel].astype(np.float64).copy()
    h0 = (channel.shape[0] // 2) * 2
    w0 = (channel.shape[1] // 2) * 2

    capacity = (h0 // 2) * (w0 // 2)
    if n * pool > capacity:
        raise ValueError(
            f"carrier_pool={pool} needs {n*pool} disjoint blocks but only {capacity} are available"
        )

    positions = selected_block_positions((h0, w0), 2, n * pool, key)
    pos = np.asarray(positions, dtype=np.int64).reshape(n, pool, 2)
    base = _gather_blocks(channel, positions).reshape(n, pool, 2, 2)
    continuous, rounded, energy, rounded_sse, feasible = _embed_candidate_bank(base, bits, period)
    selectors = _select_carriers(energy, rounded_sse, feasible, cfg.carrier_selector_policy)

    chosen_pos = pos[np.arange(n), selectors.astype(np.int64)]
    chosen_cont = _selected_from_bank(continuous, selectors)[:, None, :, :]
    chosen_round = _selected_from_bank(rounded, selectors)[:, None, :, :]
    rr, cc = chosen_pos[:, 0], chosen_pos[:, 1]
    rb = chosen_round[:, 0]
    working = channel.copy()
    working[rr, cc] = rb[:, 0, 0]
    working[rr, cc + 1] = rb[:, 0, 1]
    working[rr + 1, cc] = rb[:, 1, 0]
    working[rr + 1, cc + 1] = rb[:, 1, 1]

    out = host.copy()
    out[:h0, :w0, cfg.channel] = working[:h0, :w0].astype(np.uint8)

    cert = np.zeros(n, dtype=bool)
    cert_state = None
    if bool(cfg.compute_certificate):
        bank = convolution_bank(cfg.convolution_kernel_size, cfg.convolution_gaussian_sigmas)
        if bank:
            extremes = []
            work_final = out[:h0, :w0, cfg.channel].astype(np.float64)
            for ker in bank:
                attacked = reflect_convolve(work_final, ker.kernel)
                extremes.append(_gather_chosen(attacked, chosen_pos))
            ext = np.stack(extremes, axis=0)
        else:
            ext = np.empty((0, n, 1, 2, 2), dtype=np.float64)
        cert_cfg = replace(cfg, period_candidates=(period,), repetition=1)
        cert_state = certify_spread_groups_arrays(
            chosen_cont,
            chosen_round,
            ext,
            np.zeros(n, dtype=np.int64),
            cert_cfg,
            tighten_final=bool(cfg.certificate_final_tighten),
            embedding_feasible=np.ones(n, dtype=bool),
        )
        cert = np.asarray(cert_state["certified"], dtype=bool)

    side_metadata = {
        "method": "mecqr_qim_v2",
        "algorithm_revision": "minimum_energy_single_carrier_v2",
        "watermark_shape": [cfg.watermark_size, cfg.watermark_size],
        "block_size": 2,
        "channel": int(cfg.channel),
        "arnold_iterations": int(cfg.arnold_iterations),
        "qim_period": period,
        "carrier_pool": pool,
        "carrier_selector_policy": str(cfg.carrier_selector_policy),
        "selector_count": n,
        "key_fingerprint": hashlib.sha256(key).hexdigest()[:16],
    }
    side = build_selector_side_info(
        selectors,
        side_metadata,
        key,
        certified_mask=cert if bool(cfg.store_certificate_mask) else None,
    )
    overhead = side_info_overhead_bits(side)

    total_samples = float(host.size)
    sse = float(np.sum((out.astype(np.float64) - host.astype(np.float64)) ** 2))
    psnr_db = float("inf") if sse == 0 else 10.0 * np.log10(total_samples * 255.0**2 / sse)
    chosen_energy = energy[np.arange(n), selectors.astype(np.int64)]
    chosen_sse = rounded_sse[np.arange(n), selectors.astype(np.int64)]

    meta = {
        "method": "mecqr_qim_v2",
        "algorithm_revision": "minimum_energy_single_carrier_v2",
        "mathematical_core": "argmin delta^2 over keyed candidate carrier and exact binary-QIM feasibility",
        "payload_embeddings_per_bit": 1,
        "repetition_used": False,
        "qim_period": period,
        "decision_margin": period / 4.0,
        "carrier_pool": pool,
        "carrier_selector_policy": str(cfg.carrier_selector_policy),
        "selector_histogram": {str(j): int(np.sum(selectors == j)) for j in range(pool)},
        "continuous_embedding_energy": float(np.sum(chosen_energy)),
        "rounded_embedding_sse_selected_blocks": float(np.sum(chosen_sse)),
        "actual_rgb_sse": sse,
        "actual_psnr_db": psnr_db,
        "certificate_computed": bool(cfg.compute_certificate),
        "certificate_mode": "single_post_embedding_pass" if cfg.compute_certificate else "disabled",
        "side_information_overhead_bits": overhead,
    }
    if cfg.compute_certificate:
        meta["certified_fraction"] = float(cert.mean())
        meta["certified_groups"] = int(cert.sum())
        meta["uncertified_groups"] = int(n - cert.sum())
    if cert_state is not None:
        finite = np.asarray(cert_state["certificate_margin"], dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        meta["mean_certificate_margin"] = float(finite.mean()) if finite.size else float("-inf")
        meta["min_certificate_margin"] = float(finite.min()) if finite.size else float("-inf")
    return out, side, meta
