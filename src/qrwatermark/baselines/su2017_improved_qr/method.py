from __future__ import annotations

from typing import Any

import numpy as np

from ...core.config import BaselineConfig
from ...core.interfaces import WatermarkMethod
from ...core.types import EmbeddingResult, ExtractionResult
from ..common import canonical_qr, finish_payload, keyed_positions, prepare_payload, relation_embed


class Su2017ImprovedQR(WatermarkMethod):
    """Su et al., Multimedia Tools and Applications 76 (2017), 707-729.

    The paper's defining carrier is implemented directly: non-overlapping 3x3
    blocks and the relation between q(2,1) and q(3,1) in Q.  The authors' source
    is not public; the relative-margin update below is a transparent
    paper-aligned reconstruction of that published carrier rule, under the
    repository's common binary benchmark protocol.
    """

    name = "su2017_improved_qr"
    strength_field = "threshold"

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig(block_size=3, threshold=0.04)
        if self.config.block_size != 3:
            raise ValueError("Su2017 improved-QR baseline is defined on 3x3 blocks")
        if self.config.threshold <= 0:
            raise ValueError("threshold T must be positive")

    @staticmethod
    def _decode(qmat: np.ndarray) -> int:
        return 1 if float(qmat[1, 0]) >= float(qmat[2, 0]) else 0

    def embed(self, host: np.ndarray, watermark: np.ndarray, *, key: bytes) -> EmbeddingResult:
        c = self.config
        out = np.asarray(host, dtype=np.uint8).copy()
        bits = prepare_payload(watermark, c.arnold_iterations)
        channel = out[:, :, c.channel].astype(np.float64)
        h0 = (channel.shape[0] // 3) * 3
        w0 = (channel.shape[1] // 3) * 3
        positions = keyed_positions((h0, w0), 3, len(bits), key)

        for bit, (row, col) in zip(bits, positions):
            block = channel[row : row + 3, col : col + 3]
            qmat, rmat = canonical_qr(block)
            q2 = qmat.copy()
            q2[1, 0], q2[2, 0] = relation_embed(
                float(qmat[1, 0]), float(qmat[2, 0]), int(bit), float(c.threshold)
            )
            candidate = q2 @ rmat
            channel[row : row + 3, col : col + 3] = np.clip(np.rint(candidate), 0, 255)

        out[:h0, :w0, c.channel] = channel[:h0, :w0].astype(np.uint8)
        return EmbeddingResult(
            image=out,
            side_info=None,
            metadata={
                "reference": "Su et al. 2017, Multimedia Tools and Applications 76(1):707-729",
                "doi": "10.1007/s11042-015-3071-x",
                "carrier": "relation Q[1,0],Q[2,0]",
                "threshold": float(c.threshold),
                "implementation_status": "paper-aligned reconstruction; authors' source unavailable",
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
        h0 = (channel.shape[0] // 3) * 3
        w0 = (channel.shape[1] // 3) * 3
        count = int(watermark_shape[0] * watermark_shape[1])
        positions = keyed_positions((h0, w0), 3, count, key)
        bits = np.zeros(count, dtype=np.uint8)

        for i, (row, col) in enumerate(positions):
            block = channel[row : row + 3, col : col + 3]
            qmat, _ = canonical_qr(block)
            bits[i] = self._decode(qmat)

        return ExtractionResult(watermark=finish_payload(bits, watermark_shape, c.arnold_iterations))
