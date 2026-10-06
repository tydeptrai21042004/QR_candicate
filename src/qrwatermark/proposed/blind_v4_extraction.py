from __future__ import annotations

import numpy as np

from ..core.config import ProposedConfig
from ..utils.permutation import selected_block_arrays
from ..utils.watermark import watermark_from_scrambled_bits
from .coupled_qr import decode_coupled_selected_pixels


def extract_blind_v4_image(
    image: np.ndarray,
    key: bytes,
    cfg: ProposedConfig,
    watermark_shape: tuple[int, int] | None = None,
):
    cfg.validate()
    expected = (int(cfg.watermark_size), int(cfg.watermark_size))
    if watermark_shape is not None and tuple(map(int, watermark_shape)) != expected:
        raise ValueError(
            f"blind decoder shape is fixed by the public config as {expected}, got {watermark_shape}"
        )
    image_arr = np.asarray(image)
    if image_arr.ndim != 3 or image_arr.shape[2] < 3:
        raise ValueError("image must be color")
    channel = image_arr[:, :, cfg.channel]
    h0 = (channel.shape[0] // 2) * 2
    w0 = (channel.shape[1] // 2) * 2
    n = expected[0] * expected[1]
    rr, cc = selected_block_arrays((h0, w0), 2, n, key)

    bits, conf, detail = decode_coupled_selected_pixels(
        channel[:h0, :w0],
        rr,
        cc,
        q_period=float(cfg.coupled_q_period),
        r_period=float(cfg.coupled_r_period),
        q_weight=float(cfg.coupled_q_weight),
        r_weight=float(cfg.coupled_r_weight),
        energy_tau=float(cfg.coupled_energy_tau),
    )
    recovered = watermark_from_scrambled_bits(
        bits, int(expected[0]), int(cfg.arnold_iterations)
    )
    return recovered, float(np.mean(conf)), {
        "method": "blind_cqr_qim_v4",
        "algorithm_revision": "fully_blind_coupled_q_normalized_r12_v4",
        "fully_blind": True,
        "side_information_used": False,
        "original_host_used": False,
        "q_period": float(cfg.coupled_q_period),
        "r_period": float(cfg.coupled_r_period),
        "q_weight": float(cfg.coupled_q_weight),
        "r_weight": float(cfg.coupled_r_weight),
        "energy_tau": float(cfg.coupled_energy_tau),
        "bit_confidence_mean": float(np.mean(conf)),
        "q_confidence_mean": float(np.mean(detail["q_confidence"])),
        "r_confidence_mean": float(np.mean(detail["r_confidence"])),
        "payload_embeddings_per_bit": 1,
        "physical_observations_per_bit": 2,
        "repetition_across_blocks": False,
        "runtime_path": "vectorized_same_block_coupled_qr",
    }
