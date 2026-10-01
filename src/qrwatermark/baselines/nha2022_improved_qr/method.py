from __future__ import annotations

from typing import Any

import numpy as np

from ...core.config import BaselineConfig
from ...core.interfaces import WatermarkMethod
from ...core.types import EmbeddingResult, ExtractionResult
from ..common import finish_payload, keyed_positions, prepare_payload, qim_decode_phase, qim_embed_phase


def improved_qr_r_first(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Nha et al. R-first QR construction from A^T A = R^T R.

    This follows the recurrence in the paper's Eq. (8), then computes Q using
    Eq. (9).  For i>=2, when the value under a diagonal square root is zero or
    negative, the paper explicitly sets r_ii=1 to avoid infinities.
    """
    a = np.asarray(block, dtype=np.float64)
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("improved_qr_r_first requires a square matrix")
    n = a.shape[0]
    m = a.T @ a
    r = np.zeros((n, n), dtype=np.float64)

    # Eq. (8), first row.
    r[0, 0] = float(np.sqrt(max(float(m[0, 0]), 0.0)))
    if r[0, 0] <= 0.0:
        r[0, 0] = 1.0
    for j in range(1, n):
        r[0, j] = float(m[0, j]) / r[0, 0]

    for i in range(1, n):
        radicand = float(m[i, i] - np.dot(r[:i, i], r[:i, i]))
        r[i, i] = float(np.sqrt(radicand)) if radicand > 0.0 else 1.0
        for j in range(i + 1, n):
            numerator = float(m[i, j] - np.dot(r[:i, i], r[:i, j]))
            r[i, j] = numerator / r[i, i]

    # Eq. (9): compute Q from A and the already computed R.
    q = np.zeros_like(a, dtype=np.float64)
    q[:, 0] = a[:, 0] / r[0, 0]
    for i in range(1, n):
        residual = a[:, i].copy()
        for k in range(i):
            residual -= r[k, i] * q[:, k]
        q[:, i] = residual / r[i, i]
    return q, r


class Nha2022ImprovedQR(WatermarkMethod):
    """Nha, Thanh & Phong, Soft Computing 26 (2022), 5069-5093."""

    name = "nha2022_improved_qr"
    strength_field = "quant_step"

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig(block_size=4, quant_step=10.0)
        if self.config.block_size != 4:
            raise ValueError("Nha2022 improved QR baseline is defined on 4x4 blocks")
        if self.config.quant_step <= 0:
            raise ValueError("quant_step q must be positive")

    def embed(self, host: np.ndarray, watermark: np.ndarray, *, key: bytes) -> EmbeddingResult:
        c = self.config
        out = np.asarray(host, dtype=np.uint8).copy()
        bits = prepare_payload(watermark, c.arnold_iterations)
        channel = out[:, :, c.channel].astype(np.float64)
        h0 = (channel.shape[0] // 4) * 4
        w0 = (channel.shape[1] // 4) * 4
        positions = keyed_positions((h0, w0), 4, len(bits), key)

        for bit, (row, col) in zip(bits, positions):
            block = channel[row : row + 4, col : col + 4]
            qmat, rmat = improved_qr_r_first(block)
            r2 = rmat.copy()
            r2[0, 0] = qim_embed_phase(float(r2[0, 0]), int(bit), float(c.quant_step))
            candidate = qmat @ r2
            channel[row : row + 4, col : col + 4] = np.clip(np.rint(candidate), 0, 255)

        out[:h0, :w0, c.channel] = channel[:h0, :w0].astype(np.uint8)
        return EmbeddingResult(
            image=out,
            side_info=None,
            metadata={
                "reference": "Nha, Thanh & Phong 2022, Soft Computing 26:5069-5093",
                "doi": "10.1007/s00500-022-06975-3",
                "factorization": "R-first from A^T A = R^T R",
                "carrier": "R[0,0]",
                "quant_step": float(c.quant_step),
                "side_information_bits": 0,
            },
        )

    def extract(
        self,
        image: np.ndarray,
        *,
        key: bytes,
        side_info: dict[str, Any] | None = None,
        watermark_shape: tuple[int, int] = (64, 64),
    ) -> ExtractionResult:
        c = self.config
        channel = np.asarray(image, dtype=np.uint8)[:, :, c.channel].astype(np.float64)
        h0 = (channel.shape[0] // 4) * 4
        w0 = (channel.shape[1] // 4) * 4
        count = int(watermark_shape[0] * watermark_shape[1])
        positions = keyed_positions((h0, w0), 4, count, key)
        bits = np.zeros(count, dtype=np.uint8)

        for i, (row, col) in enumerate(positions):
            block = channel[row : row + 4, col : col + 4]
            # Paper Eq. (12): R(1,1) is just the Euclidean length of column 1.
            r11 = float(np.linalg.norm(block[:, 0]))
            bits[i] = qim_decode_phase(r11, float(c.quant_step))

        return ExtractionResult(watermark=finish_payload(bits, watermark_shape, c.arnold_iterations))
