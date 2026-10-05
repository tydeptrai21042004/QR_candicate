from __future__ import annotations

"""Fully blind deterministic-carrier QR embedding.

The decoder needs only the received image, the secret key, and the public
method configuration.  There is no selector stream, period map, certificate
mask, original host, or reference watermark.

For each payload bit one keyed 2x2 block is assigned.  The first column is
left unchanged.  If q1 is its unit direction and r12=q1^T b is the QR feature,
we solve the one-dimensional projection

    min_delta delta^2
    s.t. r12 + delta in S_bit(Delta)
         0 <= b + delta q1 <= 255,

where S_bit is a safe interval inside the corresponding hard-decision half of
one QIM period.  With the default safe-margin ratio 1/8, the two sets are

    S0 = U_k Delta [k+1/8, k+3/8]
    S1 = U_k Delta [k+5/8, k+7/8].

This is a single Euclidean projection, not a collection of repair rules.
"""

from dataclasses import replace

import numpy as np

from ..core.config import ProposedConfig
from ..utils.permutation import selected_block_arrays
from ..utils.watermark import scrambled_bits_from_watermark
from .convolution import convolution_bank, reflect_convolve
from .elegant_embedding import _candidate_geometry, _gather_blocks
from .qr_sensitivity import (
    convex_hull_group_bound_batch,
    tightened_two_extreme_convex_group_bound_batch,
    tightened_two_extreme_single_carrier_bound_batch,
)
from .spread_qim import unit_spread_weights
from .streaming_datapath import project_safe_selected_pixels, selected_r12_float


def _safe_geometry(period: float, margin_ratio: float) -> tuple[float, float]:
    period = float(period)
    ratio = float(margin_ratio)
    if period <= 0.0:
        raise ValueError("qim_period must be positive")
    if not (0.0 < ratio < 0.25):
        raise ValueError("blind_margin_ratio must lie strictly in (0, 0.25)")
    rho = ratio * period
    half_width = 0.25 * period - rho
    return rho, half_width


def _project_safe_intervals(
    base: np.ndarray,
    bits: np.ndarray,
    period: float,
    margin_ratio: float,
):
    """Project fixed carriers onto the nearest feasible safe decision interval."""
    blocks = np.asarray(base, dtype=np.float64)
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if blocks.shape != (bits.size, 2, 2):
        raise ValueError("base must have shape (n_bits,2,2)")

    q1, r12, lo, hi = _candidate_geometry(blocks[:, None, :, :])
    q1 = q1[:, 0]
    r12 = r12[:, 0]
    lo = lo[:, 0]
    hi = hi[:, 0]

    rho, half_width = _safe_geometry(period, margin_ratio)
    center = (0.25 + 0.50 * bits.astype(np.float64)) * float(period)
    feasible_lo = r12 + lo
    feasible_hi = r12 + hi

    # The target set is a periodic union of closed intervals.  The nearest
    # feasible interval index is the nearest unconstrained index clipped to the
    # exact set of interval indices that intersect the pixel-feasible segment.
    k_near = np.rint((r12 - center) / float(period))
    k_min = np.ceil(
        (feasible_lo - center - half_width) / float(period) - 1e-12
    )
    k_max = np.floor(
        (feasible_hi - center + half_width) / float(period) + 1e-12
    )
    feasible = k_min <= k_max
    if not np.all(feasible):
        count = int(np.sum(~feasible))
        raise ValueError(
            f"{count} keyed carriers cannot realize the requested blind safe set; "
            "reduce qim_period or blind_margin_ratio"
        )

    k = np.minimum(np.maximum(k_near, k_min), k_max)
    interval_lo = np.maximum(center + k * float(period) - half_width, feasible_lo)
    interval_hi = np.minimum(center + k * float(period) + half_width, feasible_hi)
    target = np.minimum(np.maximum(r12, interval_lo), interval_hi)
    delta = target - r12

    continuous = blocks.copy()
    continuous[:, :, 1] += delta[:, None] * q1
    rounded = np.clip(np.rint(continuous), 0, 255).astype(np.float64)

    # For 2x2 blocks, rounding the modified column changes r12 by at most
    # ||e||_2 <= sqrt(2)/2 because q1 is unit length and the first column is
    # unchanged.  The explicit check protects nonstandard parameter choices.
    _, rounded_r12, _, _ = _candidate_geometry(rounded[:, None, :, :])
    phase = np.mod(rounded_r12[:, 0], float(period))
    decoded = (phase >= 0.5 * float(period)).astype(np.uint8)
    decode_ok = decoded == bits
    if not np.all(decode_ok):
        count = int(np.sum(~decode_ok))
        raise ValueError(
            f"{count} carriers lost their bit under uint8 rounding; increase "
            "blind_margin_ratio or qim_period"
        )

    return continuous, rounded, delta, rho


def _project_center_lattice(base: np.ndarray, bits: np.ndarray, period: float):
    """Blind fixed-carrier point-QIM ablation using the old quarter centers."""
    blocks = np.asarray(base, dtype=np.float64)
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    q1, r12, lo, hi = _candidate_geometry(blocks[:, None, :, :])
    q1 = q1[:, 0]
    r12 = r12[:, 0]
    lo = lo[:, 0]
    hi = hi[:, 0]
    offsets = (0.25 + 0.50 * bits.astype(np.float64)) * float(period)
    target_lo = r12 + lo
    target_hi = r12 + hi
    k_near = np.rint((r12 - offsets) / float(period))
    k_min = np.ceil((target_lo - offsets) / float(period) - 1e-12)
    k_max = np.floor((target_hi - offsets) / float(period) + 1e-12)
    feasible = k_min <= k_max
    if not np.all(feasible):
        raise ValueError("fixed-carrier center-QIM is infeasible for at least one bit")
    k = np.minimum(np.maximum(k_near, k_min), k_max)
    target = offsets + k * float(period)
    delta = target - r12
    continuous = blocks.copy()
    continuous[:, :, 1] += delta[:, None] * q1
    rounded = np.clip(np.rint(continuous), 0, 255).astype(np.float64)
    _, rr12, _, _ = _candidate_geometry(rounded[:, None, :, :])
    decoded = (np.mod(rr12[:, 0], float(period)) >= 0.5 * float(period)).astype(np.uint8)
    if not np.all(decoded == bits):
        raise ValueError("fixed-carrier center-QIM lost a bit under uint8 rounding")
    return continuous, rounded, delta, 0.25 * float(period)


def _gather_fixed(channel: np.ndarray, chosen_positions: np.ndarray) -> np.ndarray:
    rr, cc = chosen_positions[:, 0], chosen_positions[:, 1]
    x = np.asarray(channel, dtype=np.float64)
    out = np.empty((chosen_positions.shape[0], 1, 2, 2), dtype=np.float64)
    out[:, 0, 0, 0] = x[rr, cc]
    out[:, 0, 0, 1] = x[rr, cc + 1]
    out[:, 0, 1, 0] = x[rr + 1, cc]
    out[:, 0, 1, 1] = x[rr + 1, cc + 1]
    return out


def _blind_certificate(
    out: np.ndarray,
    chosen_positions: np.ndarray,
    period: float,
    cfg: ProposedConfig,
):
    """Post-embedding certificate from the actual rounded blind codeword.

    The certificate is reporting only.  It is never serialized and is not used
    by extraction.  The available hard-decision margin of each rounded carrier
    is compared directly with the convolution theorem bound.
    """
    work = out[:, :, cfg.channel].astype(np.float64)
    rounded = _gather_fixed(work, chosen_positions)
    rr = chosen_positions[:, 0]
    cc = chosen_positions[:, 1]
    # Closed-form canonical r12 is exact for 2x2 blocks and avoids thousands
    # of scalar np.linalg.qr fallbacks on dark/degenerate first columns.
    vals = selected_r12_float(work, rr, cc)
    phase = np.mod(vals, float(period))
    decision_margin = np.minimum(
        np.minimum(phase, np.abs(phase - 0.5 * float(period))),
        float(period) - phase,
    )

    bank = convolution_bank(cfg.convolution_kernel_size, cfg.convolution_gaussian_sigmas)
    if bank:
        extremes = []
        for ker in bank:
            attacked = reflect_convolve(work, ker.kernel)
            extremes.append(_gather_fixed(attacked, chosen_positions))
        ext = np.stack(extremes, axis=0)
        w = unit_spread_weights(1)
        if bool(cfg.certificate_final_tighten) and ext.shape[0] == 2:
            # blind-v3 has exactly one carrier per bit.  Use the algebraically
            # identical repetition=1 theorem kernel to avoid generic singleton
            # reductions in the per-frame certificate path.
            conv, observed, extreme, fallback, use_path, intervals = (
                tightened_two_extreme_single_carrier_bound_batch(
                    rounded[:, 0],
                    ext[:, :, 0],
                    subdivisions=int(cfg.certificate_path_subdivisions),
                )
            )
        else:
            conv, observed, extreme = convex_hull_group_bound_batch(rounded, ext, w)
            fallback = np.full(vals.size, np.inf, dtype=np.float64)
            use_path = np.zeros(vals.size, dtype=bool)
            intervals = np.empty((0, vals.size), dtype=np.float64)
    else:
        conv = np.zeros(vals.size, dtype=np.float64)
        observed = np.zeros(vals.size, dtype=np.float64)
        extreme = np.empty((0, vals.size), dtype=np.float64)
        fallback = np.full(vals.size, np.inf, dtype=np.float64)
        use_path = np.zeros(vals.size, dtype=bool)
        intervals = np.empty((0, vals.size), dtype=np.float64)

    required = (
        float(cfg.convolution_safety_factor) * np.asarray(conv, dtype=np.float64)
        + float(cfg.additive_feature_budget)
    )
    cert_margin = decision_margin - required
    certified = np.isfinite(required) & (cert_margin >= 0.0)
    return {
        "decision_margin": decision_margin,
        "convolution_bound": np.asarray(conv, dtype=np.float64),
        "convolution_shift": np.asarray(observed, dtype=np.float64),
        "extreme_bounds": np.asarray(extreme, dtype=np.float64),
        "fallback_bound": np.asarray(fallback, dtype=np.float64),
        "use_path": np.asarray(use_path, dtype=bool),
        "path_interval_bounds": np.asarray(intervals, dtype=np.float64),
        "required_margin": required,
        "certificate_margin": cert_margin,
        "certified": certified,
    }


def embed_blind_image(host, watermark, key: bytes, cfg: ProposedConfig):
    """Embed one blind-v3 payload using the RT100 selected-pixel datapath.

    The mathematical embedding is unchanged.  Runtime is reduced by keeping
    geometry/key descriptors in the session cache and touching only the 4096
    selected 2x2 blocks instead of repeatedly converting/scanning the complete
    RGB frame.  The optional robustness certificate remains an offline audit
    and is executed only when ``compute_certificate`` is enabled.
    """
    cfg.validate()
    host = np.asarray(host, dtype=np.uint8)
    wm = np.asarray(watermark, dtype=np.uint8)
    if host.ndim != 3 or host.shape[2] < 3:
        raise ValueError("host must be color")
    if wm.shape != (cfg.watermark_size, cfg.watermark_size):
        raise ValueError(f"watermark must be {cfg.watermark_size}x{cfg.watermark_size}")

    # Arnold scrambling is a fixed index permutation.  Apply it directly in
    # the bit domain so the online datapath does not allocate/permutate a 2-D
    # image.  The result is bit-for-bit identical to the original path.
    bits = scrambled_bits_from_watermark(wm, cfg.arnold_iterations)
    n = int(bits.size)
    period = float(cfg.qim_period)
    channel = host[:, :, cfg.channel]
    h0 = (channel.shape[0] // 2) * 2
    w0 = (channel.shape[1] // 2) * 2
    capacity = (h0 // 2) * (w0 // 2)
    if n > capacity:
        raise ValueError(f"payload needs {n} blocks but only {capacity} are available")

    # Key-only carrier geometry is a control-plane object.  The cached row and
    # column vectors are identical to the legacy HMAC permutation but avoid
    # rebuilding thousands of Python tuples for every video frame.
    rr, cc = selected_block_arrays((h0, w0), 2, n, key)

    if cfg.blind_projection == "safe_set":
        # FPGA-oriented hot path: four selected samples enter the scalar 2x2
        # datapath and only the second column is written back.  This is the
        # exact same QR safe-set projection as the matrix reference path.
        y0, y1, delta, nominal_margin, sse, changed_samples = project_safe_selected_pixels(
            channel, rr, cc, bits, period, cfg.blind_margin_ratio
        )
        mathematical_core = (
            "Euclidean projection of r12 onto the nearest pixel-feasible blind safe-decision interval"
        )
        out = host.copy()
        out_channel = out[:, :, cfg.channel]
        out_channel[rr, cc + 1] = y0
        out_channel[rr + 1, cc + 1] = y1
    elif cfg.blind_projection == "center":
        # The center-QIM branch is an ablation only; keep the existing generic
        # reference implementation rather than complicating the FPGA datapath.
        base = np.empty((n, 2, 2), dtype=np.float64)
        base[:, 0, 0] = channel[rr, cc]
        base[:, 0, 1] = channel[rr, cc + 1]
        base[:, 1, 0] = channel[rr + 1, cc]
        base[:, 1, 1] = channel[rr + 1, cc + 1]
        _continuous, rounded, delta, nominal_margin = _project_center_lattice(base, bits, period)
        mathematical_core = "fixed-carrier quarter-coset QIM center projection"
        out = host.copy()
        out_channel = out[:, :, cfg.channel]
        out_channel[rr, cc + 1] = rounded[:, 0, 1].astype(np.uint8, copy=False)
        out_channel[rr + 1, cc + 1] = rounded[:, 1, 1].astype(np.uint8, copy=False)
        diff = rounded - base
        sse = float(np.sum(diff * diff))
        changed_samples = int(np.count_nonzero(rounded != base))
    else:
        raise ValueError(f"unknown blind_projection={cfg.blind_projection!r}")

    cert_state = None
    if bool(cfg.compute_certificate):
        # Certificate analysis is deliberately outside the online RT path.  It
        # is report-only and never affects image formation or decoding.
        chosen_pos = np.column_stack((rr, cc)).astype(np.int64, copy=False)
        cert_state = _blind_certificate(out[:h0, :w0], chosen_pos, period, cfg)

    # Only the selected second-column samples can differ.  ``sse`` and
    # ``changed_samples`` above are exact, so no full-frame scan is required.
    total_samples = float(host.size)
    psnr_db = float("inf") if sse == 0 else 10.0 * np.log10(total_samples * 255.0**2 / sse)

    zero_overhead = {
        "period_code_bits": 0,
        "selector_code_bits": 0,
        "certificate_mask_bits": 0,
        "authentication_bits": 0,
        "logical_total_bits": 0,
        "total_serialized_bits": 0,
    }
    meta = {
        "method": "blind_mecqr_qim_v3",
        "algorithm_revision": "fully_blind_safe_projection_v3_streaming_exact",
        "mathematical_core": mathematical_core,
        "fully_blind": True,
        "requires_original_host": False,
        "requires_side_information": False,
        "side_information_bits": 0,
        "side_information_overhead_bits": zero_overhead,
        "payload_embeddings_per_bit": 1,
        "repetition_used": False,
        "carrier_mapping": "key_only_deterministic_one_to_one",
        "qim_period": period,
        "blind_projection": str(cfg.blind_projection),
        "blind_margin_ratio": float(cfg.blind_margin_ratio),
        "nominal_safe_margin": float(nominal_margin),
        "continuous_embedding_energy": float(np.sum(delta * delta)),
        "actual_rgb_sse": sse,
        "actual_psnr_db": psnr_db,
        "changed_samples": changed_samples,
        "certificate_computed": bool(cfg.compute_certificate),
        "certificate_mode": "offline_report_only_no_side_info" if cfg.compute_certificate else "disabled_online_rt_path",
        "runtime_path": "unified_streaming_2x2_second_column_only",
    }
    if cert_state is not None:
        cert = np.asarray(cert_state["certified"], dtype=bool)
        margins = np.asarray(cert_state["certificate_margin"], dtype=np.float64)
        finite = margins[np.isfinite(margins)]
        meta.update(
            {
                "certified_fraction": float(np.mean(cert)) if cert.size else 0.0,
                "certified_groups": int(np.sum(cert)),
                "uncertified_groups": int(cert.size - np.sum(cert)),
                "mean_certificate_margin": float(np.mean(finite)) if finite.size else float("-inf"),
                "min_certificate_margin": float(np.min(finite)) if finite.size else float("-inf"),
            }
        )

    return out, None, meta
