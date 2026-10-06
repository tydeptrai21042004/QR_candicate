from __future__ import annotations

import numpy as np

from ..core.config import ProposedConfig
from ..utils.permutation import selected_block_arrays
from ..utils.watermark import scrambled_bits_from_watermark
from .coupled_qr import project_coupled_selected_pixels


def embed_blind_v4_image(host, watermark, key: bytes, cfg: ProposedConfig):
    """Fully blind coupled-Q/R embedding with one keyed block per payload bit."""
    cfg.validate()
    host = np.asarray(host, dtype=np.uint8)
    wm = np.asarray(watermark, dtype=np.uint8)
    if host.ndim != 3 or host.shape[2] < 3:
        raise ValueError("host must be color")
    if wm.shape != (cfg.watermark_size, cfg.watermark_size):
        raise ValueError(f"watermark must be {cfg.watermark_size}x{cfg.watermark_size}")

    bits = scrambled_bits_from_watermark(wm, cfg.arnold_iterations)
    n = int(bits.size)
    channel = host[:, :, cfg.channel]
    h0 = (channel.shape[0] // 2) * 2
    w0 = (channel.shape[1] // 2) * 2
    capacity = (h0 // 2) * (w0 // 2)
    if n > capacity:
        raise ValueError(f"payload needs {n} blocks but only {capacity} are available")
    rr, cc = selected_block_arrays((h0, w0), 2, n, key)

    state = project_coupled_selected_pixels(
        channel,
        rr,
        cc,
        bits,
        q_period=float(cfg.coupled_q_period),
        r_period=float(cfg.coupled_r_period),
        q_margin_ratio=float(cfg.coupled_q_margin_ratio),
        r_margin_ratio=float(cfg.coupled_r_margin_ratio),
    )

    out = host.copy()
    y = out[:, :, cfg.channel]
    y[rr, cc] = state["a0"]
    y[rr + 1, cc] = state["a1"]
    y[rr, cc + 1] = state["b0"]
    y[rr + 1, cc + 1] = state["b1"]

    total_samples = float(host.size)
    sse = float(state["sse"])
    psnr_db = float("inf") if sse <= 0.0 else float(
        10.0 * np.log10(total_samples * 255.0**2 / sse)
    )
    zero_overhead = {
        "period_code_bits": 0,
        "selector_code_bits": 0,
        "certificate_mask_bits": 0,
        "authentication_bits": 0,
        "logical_total_bits": 0,
        "total_serialized_bits": 0,
    }
    meta = {
        "method": "blind_cqr_qim_v4",
        "algorithm_revision": "fully_blind_coupled_q_normalized_r12_v4",
        "mathematical_core": (
            "same-block coupled Q-direction QIM plus scale-normalized QR r12 QIM; "
            "closed-form r12 update b'=b+(r12'-r12)q1"
        ),
        "normalized_r_carrier": "asinh(r12/sqrt(r11^2+r22^2+1))",
        "fully_blind": True,
        "requires_original_host": False,
        "requires_side_information": False,
        "side_information_bits": 0,
        "side_information_overhead_bits": zero_overhead,
        "payload_bits": n,
        "payload_embeddings_per_bit": 1,
        "physical_observations_per_bit": 2,
        "repetition_across_blocks": False,
        "q_period": float(cfg.coupled_q_period),
        "r_period": float(cfg.coupled_r_period),
        "q_margin_ratio": float(cfg.coupled_q_margin_ratio),
        "r_margin_ratio": float(cfg.coupled_r_margin_ratio),
        "q_projection_feasible_fraction": float(np.mean(state["q_projection_feasible"])),
        "r_projection_feasible_fraction": float(np.mean(state["r_projection_feasible"])),
        "q_integer_match_fraction": float(np.mean(state["q_clean_match"])),
        "r_integer_match_fraction": float(np.mean(state["r_clean_match"])),
        "q_integer_closure_count": int(state.get("q_integer_closure_count", 0)),
        "r_integer_closure_count": int(state.get("r_integer_closure_count", 0)),
        "q_confidence_mean": float(np.mean(state["q_confidence"])),
        "r_confidence_mean": float(np.mean(state["r_confidence"])),
        "changed_channel_samples": int(state["changed_samples"]),
        "embedding_sse": sse,
        "embedding_psnr_from_sparse_sse": psnr_db,
        "runtime_path": "vectorized_same_block_coupled_qr",
    }
    return out, None, meta
