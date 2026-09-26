from __future__ import annotations

from typing import Any

import numpy as np

from ...core.config import BaselineConfig
from ...core.interfaces import WatermarkMethod
from ...core.types import EmbeddingResult, ExtractionResult
from ..common import finish_payload, keyed_positions, prepare_payload, qim_decode_phase, qim_embed_phase


class Nha2022ImprovedQR(WatermarkMethod):
    """Reference implementation of Nha, Thanh & Phong (Soft Computing, 2022).

    The published rule embeds one bit in R(1,1) of each 4x4 blue-channel block
    using quarter-period QIM. Extraction obtains R(1,1) as the Euclidean norm
    of the first column, avoiding a QR decomposition at decode time.
    """

    name = "nha2022_improved_qr"

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig(block_size=4, quant_step=8.0)

    def embed(self, host: np.ndarray, watermark: np.ndarray, *, key: bytes) -> EmbeddingResult:
        c = self.config
        out = np.asarray(host, dtype=np.uint8).copy()
        bits = prepare_payload(watermark, c.arnold_iterations)
        channel = out[:, :, c.channel].astype(np.float64)
        h0 = (channel.shape[0] // c.block_size) * c.block_size
        w0 = (channel.shape[1] // c.block_size) * c.block_size
        pos = keyed_positions((h0, w0), c.block_size, len(bits), key)
        for bit, (row, col) in zip(bits, pos):
            block = channel[row : row + c.block_size, col : col + c.block_size]
            qmat, rmat = np.linalg.qr(block)
            signs = np.sign(np.diag(rmat)); signs[signs == 0] = 1.0
            d = np.diag(signs); qmat = qmat @ d; rmat = d @ rmat
            r2 = rmat.copy()
            r2[0, 0] = qim_embed_phase(float(r2[0, 0]), int(bit), c.quant_step)
            cand = qmat @ r2
            channel[row : row + c.block_size, col : col + c.block_size] = np.clip(np.rint(cand), 0, 255)
        out[:h0, :w0, c.channel] = channel[:h0, :w0].astype(np.uint8)
        return EmbeddingResult(image=out, side_info=None, metadata={"reference": "Nha et al. 2022", "q": c.quant_step})

    def extract(self, image: np.ndarray, *, key: bytes, side_info: dict[str, Any] | None = None, watermark_shape=(64, 64)) -> ExtractionResult:
        c = self.config
        channel = np.asarray(image, dtype=np.uint8)[:, :, c.channel].astype(np.float64)
        h0 = (channel.shape[0] // c.block_size) * c.block_size
        w0 = (channel.shape[1] // c.block_size) * c.block_size
        l = int(watermark_shape[0] * watermark_shape[1])
        pos = keyed_positions((h0, w0), c.block_size, l, key)
        bits = np.zeros(l, dtype=np.uint8)
        for i, (row, col) in enumerate(pos):
            block = channel[row : row + c.block_size, col : col + c.block_size]
            r11 = float(np.linalg.norm(block[:, 0]))
            bits[i] = qim_decode_phase(r11, c.quant_step)
        wm = finish_payload(bits, watermark_shape, c.arnold_iterations)
        return ExtractionResult(watermark=wm)
