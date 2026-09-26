from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import hessenberg

from ...core.config import BaselineConfig
from ...core.interfaces import WatermarkMethod
from ...core.types import EmbeddingResult, ExtractionResult
from ..common import finish_payload, keyed_positions, prepare_payload, qim_decode_phase, qim_embed_phase


class Su2017Hessenberg(WatermarkMethod):
    """Reconstructed reference implementation for Su & Chen (AEU, 2017).

    The accessible paper description specifies 4x4 blocks, Hessenberg transform,
    blind quantization in the largest-energy Hessenberg coefficient, Arnold
    scrambling, and keyed block selection. This module implements those stated
    mechanisms with quarter-period QIM under the repository's unified binary
    payload protocol. It is not claimed to be the authors' original source code.
    """

    name = "su2017_hessenberg"

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig(block_size=4, quant_step=8.0)

    @staticmethod
    def _energy_index(h: np.ndarray) -> tuple[int, int]:
        mask = np.fromfunction(lambda i, j: i <= j + 1, h.shape, dtype=int).astype(bool)
        values = np.where(mask, np.abs(h), -np.inf)
        return tuple(int(x) for x in np.unravel_index(np.argmax(values), h.shape))

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
            hmat, qmat = hessenberg(block, calc_q=True)
            idx = self._energy_index(hmat)
            h2 = hmat.copy()
            sign = 1.0 if h2[idx] >= 0 else -1.0
            mag = abs(float(h2[idx]))
            h2[idx] = sign * qim_embed_phase(mag, int(bit), c.quant_step)
            cand = qmat @ h2 @ qmat.T
            channel[row : row + c.block_size, col : col + c.block_size] = np.clip(np.rint(cand), 0, 255)
        out[:h0, :w0, c.channel] = channel[:h0, :w0].astype(np.uint8)
        return EmbeddingResult(image=out, metadata={"reference": "Su & Chen 2017", "q": c.quant_step})

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
            hmat = hessenberg(block, calc_q=False)
            idx = self._energy_index(hmat)
            bits[i] = qim_decode_phase(abs(float(hmat[idx])), c.quant_step)
        return ExtractionResult(watermark=finish_payload(bits, watermark_shape, c.arnold_iterations))
