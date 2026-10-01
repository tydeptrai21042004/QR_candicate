from __future__ import annotations

from typing import Any

import numpy as np

from ...core.config import BaselineConfig
from ...core.interfaces import WatermarkMethod
from ...core.types import EmbeddingResult, ExtractionResult
from ..common import canonical_qr, finish_payload, keyed_positions, prepare_payload


class Su2014QR(WatermarkMethod):
    """Su et al., Signal Processing 94 (2014), 219-235.

    Paper rule retained exactly at the carrier level:
      * non-overlapping 4x4 blocks;
      * QR factorization;
      * quantize r(1,4) using Eqs. (22)-(26);
      * decode with mod(ceil(r(1,4)/Delta), 2), Eq. (28).

    The repository-wide binary payload, Arnold scrambling and HMAC block order
    are used so that every baseline receives exactly the same payload and keying
    protocol.  The original paper encoded a 24-bit color watermark across RGB.
    """

    name = "su2014_qr"
    strength_field = "quant_step"

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig(block_size=4, quant_step=42.0)
        if self.config.block_size != 4:
            raise ValueError("Su2014 QR baseline is defined on 4x4 blocks")
        if self.config.quant_step <= 0:
            raise ValueError("quant_step (Delta) must be positive")

    @staticmethod
    def _embed_r14(value: float, bit: int, delta: float) -> float:
        # Su et al. 2014, Eqs. (22)-(26).
        if int(bit) == 1:
            t1, t2 = 0.5 * delta, -1.5 * delta
        else:
            t1, t2 = -0.5 * delta, 1.5 * delta
        k = np.floor(np.ceil(float(value) / delta) / 2.0)
        c1 = 2.0 * k * delta + t1
        c2 = 2.0 * k * delta + t2
        return float(c2 if abs(float(value) - c2) < abs(float(value) - c1) else c1)

    @staticmethod
    def _decode_r14(value: float, delta: float) -> int:
        # Eq. (28). Python's modulo returns 0/1 for the integer here, including
        # negative ceil values, which is exactly the desired parity operation.
        return int(int(np.ceil(float(value) / delta)) % 2)

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
            qmat, rmat = canonical_qr(block)
            r2 = rmat.copy()
            r2[0, 3] = self._embed_r14(float(r2[0, 3]), int(bit), float(c.quant_step))
            candidate = qmat @ r2
            channel[row : row + 4, col : col + 4] = np.clip(np.rint(candidate), 0, 255)

        out[:h0, :w0, c.channel] = channel[:h0, :w0].astype(np.uint8)
        return EmbeddingResult(
            image=out,
            side_info=None,
            metadata={
                "reference": "Su et al. 2014, Signal Processing 94:219-235",
                "doi": "10.1016/j.sigpro.2013.06.025",
                "carrier": "R[0,3]",
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
            _, rmat = canonical_qr(block)
            bits[i] = self._decode_r14(float(rmat[0, 3]), float(c.quant_step))

        return ExtractionResult(watermark=finish_payload(bits, watermark_shape, c.arnold_iterations))
