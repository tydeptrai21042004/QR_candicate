from __future__ import annotations

import hashlib

import numpy as np

from ..core.config import ProposedConfig
from ..utils.permutation import selected_block_positions
from ..utils.watermark import arnold_transform, bits_from_watermark
from .certificate import PeriodDecision, choose_period_for_group
from .convolution import convolution_bank, circular_convolve, spectral_distance_from_identity
from .r_branch import build_r_candidate, r12_value
from .side_info import build_side_info


def _group_r12(channel: np.ndarray, positions: list[tuple[int, int]], block_size: int) -> np.ndarray:
    return np.asarray(
        [r12_value(channel[r : r + block_size, c : c + block_size]) for r, c in positions],
        dtype=np.float64,
    )


def embed_image(host: np.ndarray, watermark: np.ndarray, key: bytes, cfg: ProposedConfig):
    """Embed with convolution-certified, payload-independent R12-QIM periods."""
    cfg.validate()
    host = np.asarray(host, dtype=np.uint8)
    if host.ndim != 3 or host.shape[2] < 3:
        raise ValueError("host must be a color image")
    wm = np.asarray(watermark, dtype=np.uint8)
    if wm.shape != (cfg.watermark_size, cfg.watermark_size):
        raise ValueError(f"watermark must be {cfg.watermark_size}x{cfg.watermark_size}")

    scrambled = arnold_transform(wm, cfg.arnold_iterations)
    bits = bits_from_watermark(scrambled)
    payload_bits = int(bits.size)

    channel = host[:, :, cfg.channel].astype(np.float64).copy()
    h0 = (channel.shape[0] // cfg.block_size) * cfg.block_size
    w0 = (channel.shape[1] // cfg.block_size) * cfg.block_size
    work = channel[:h0, :w0]

    count = payload_bits * cfg.repetition
    positions = selected_block_positions((h0, w0), cfg.block_size, count, key)

    # Convolution is part of the method: the encoder calibrates local R12
    # sensitivity against a finite normalized convolution family on Z_M x Z_N.
    bank = convolution_bank(cfg.convolution_kernel_size, cfg.convolution_gaussian_sigmas)
    convolved_channels = [circular_convolve(work, item.kernel) for item in bank]
    spectral_eta = {
        item.name: spectral_distance_from_identity(item.kernel, work.shape)
        for item in bank
    }

    period_indices = np.zeros(payload_bits, dtype=np.uint8)
    decisions: list[PeriodDecision] = []

    # Crucially, this complete pass happens before the encoder reads bits[k].
    # Period selection therefore depends on host/convolution geometry, not the
    # watermark payload.
    for k in range(payload_bits):
        group_pos = positions[k * cfg.repetition : (k + 1) * cfg.repetition]
        group_blocks = [work[r : r + cfg.block_size, c : c + cfg.block_size].copy() for r, c in group_pos]
        base_r12 = np.asarray([r12_value(b) for b in group_blocks], dtype=np.float64)
        if convolved_channels:
            conv_r12 = np.vstack([_group_r12(ch, group_pos, cfg.block_size) for ch in convolved_channels])
        else:
            conv_r12 = np.empty((0, cfg.repetition), dtype=np.float64)
        decision = choose_period_for_group(base_r12, conv_r12, cfg, blocks=group_blocks)
        period_indices[k] = np.uint8(decision.period_index)
        decisions.append(decision)

    clipping_events = 0
    for k, bit in enumerate(bits):
        period = float(cfg.period_candidates[int(period_indices[k])])
        for j in range(cfg.repetition):
            row, col = positions[k * cfg.repetition + j]
            block = channel[row : row + cfg.block_size, col : col + cfg.block_size].copy()
            candidate = build_r_candidate(block, int(bit), period)
            rounded = np.rint(candidate)
            clipping_events += int(np.any((rounded < 0.0) | (rounded > 255.0)))
            channel[row : row + cfg.block_size, col : col + cfg.block_size] = np.clip(rounded, 0, 255)

    out = host.copy()
    out[:h0, :w0, cfg.channel] = channel[:h0, :w0].astype(np.uint8)

    period_table = [float(x) for x in cfg.period_candidates]
    metadata = {
        "method": "ccqr_r12_qim_v1",
        "watermark_shape": [cfg.watermark_size, cfg.watermark_size],
        "repetition": cfg.repetition,
        "block_size": cfg.block_size,
        "channel": cfg.channel,
        "arnold_iterations": cfg.arnold_iterations,
        "period_candidates": period_table,
        "period_count": len(period_table),
        "convolution_kernel_size": cfg.convolution_kernel_size,
        "convolution_gaussian_sigmas": [float(x) for x in cfg.convolution_gaussian_sigmas],
        "convolution_spectral_eta": spectral_eta,
        "key_fingerprint": hashlib.sha256(key).hexdigest()[:16],
    }
    side = build_side_info(period_indices, metadata, key)

    certified = np.asarray([d.certified for d in decisions], dtype=bool)
    range_feasible = np.asarray([d.range_feasible for d in decisions], dtype=bool)
    residuals = np.asarray([d.certificate_margin for d in decisions], dtype=np.float64)
    shifts = np.asarray([d.convolution_shift for d in decisions], dtype=np.float64)
    required = np.asarray([d.required_margin for d in decisions], dtype=np.float64)
    theoretical_mse = np.asarray([d.worst_case_mse for d in decisions], dtype=np.float64)
    hist = {str(period_table[i]): int(np.sum(period_indices == i)) for i in range(len(period_table))}

    return out, side, {
        "method": "ccqr_r12_qim_v1",
        "certified_fraction": float(certified.mean()),
        "range_feasible_fraction": float(range_feasible.mean()),
        "mean_certificate_margin": float(residuals.mean()),
        "min_certificate_margin": float(residuals.min()),
        "mean_convolution_r12_shift": float(shifts.mean()),
        "max_convolution_r12_shift": float(shifts.max()),
        "mean_required_margin": float(required.mean()),
        "mean_worst_hypothetical_block_mse": float(theoretical_mse.mean()),
        "period_histogram": hist,
        "clipping_events": int(clipping_events),
        "convolution_spectral_eta": spectral_eta,
    }
