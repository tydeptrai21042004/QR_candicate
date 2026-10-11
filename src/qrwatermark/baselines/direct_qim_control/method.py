"""Reproducible scalar pixel-QIM CONTROL, not a claim of published SOTA.

One green sample per 2x2 cell; four bits per selected 4x4 patch.
Uses the same keyed carrier positions, payload and capacity as Green-Quad.
This intentionally measures the gain from distributing the perturbation over
four pixels rather than changing all other aspects of the protocol.
"""
from __future__ import annotations
import numpy as np
from ...core.config import BaselineConfig
from ...core.interfaces import WatermarkMethod
from ...core.types import EmbeddingResult, ExtractionResult
from ...proposed.block_carrier import keyed_blocks
from ...utils.watermark import scrambled_bits_from_watermark, watermark_from_scrambled_bits


def codebook(step: int, bit: int) -> np.ndarray:
    if step < 4 or step % 2:
        raise ValueError('QIM step must be even and >=4')
    base = step // 4 + (step // 2) * int(bit)
    return np.arange(base, 256, step, dtype=np.int16)


def nearest_code(values: np.ndarray, step: int, bit: int) -> np.ndarray:
    cb = codebook(step, bit)
    if not len(cb):
        raise ValueError('QIM step is too large for uint8 dynamic range')
    idx = np.abs(values.astype(np.int16)[:,None] - cb[None,:]).argmin(axis=1)
    return cb[idx].astype(np.uint8)


class DirectPixelQIM(WatermarkMethod):
    name = 'direct_pixel_qim_control'
    strength_field = 'quant_step'

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig(quant_step=66.0)

    def _step(self):
        return min(252, max(4, 2 * round(float(self.config.quant_step) / 2)))

    def embed(self, host, watermark, *, key):
        host = np.asarray(host)
        watermark = np.asarray(watermark)
        if host.dtype != np.uint8 or host.ndim != 3 or host.shape[2] != 3:
            raise ValueError('Expected uint8 BGR host')
        if watermark.shape != (self.config.watermark_size,)*2:
            raise ValueError('Payload shape does not match method configuration')
        bits = scrambled_bits_from_watermark(watermark, self.config.arnold_iterations)
        if bits.size % 4:
            raise ValueError('Watermark length must be divisible by four')
        rr, cc = keyed_blocks(host.shape[:2], bits.size // 4, key)
        br = np.repeat(rr, 4) + np.tile([0,0,2,2], len(rr))
        bc = np.repeat(cc, 4) + np.tile([0,2,0,2], len(cc))
        out = host.copy()
        current = out[br,bc,1]
        step = self._step()
        for bit in (0,1):
            mask = bits == bit
            if np.any(mask):
                out[br[mask],bc[mask],1] = nearest_code(current[mask], step, bit)
        return EmbeddingResult(out, None, {'fully_blind':True,'side_information_bits':0,'step':step,'control_not_published_paper':True})

    def extract(self,image,*,key,side_info=None,watermark_shape=(64,64)):
        if side_info is not None:
            raise ValueError('Direct QIM is blind')
        n = self.config.watermark_size
        if watermark_shape != (n,n):
            raise ValueError('Expected configured watermark size')
        rr, cc = keyed_blocks(image.shape[:2], n*n//4, key)
        br = np.repeat(rr,4)+np.tile([0,0,2,2],len(rr))
        bc = np.repeat(cc,4)+np.tile([0,2,0,2],len(cc))
        values=np.asarray(image)[br,bc,1].astype(np.int16)
        step=self._step()
        c0,c1=codebook(step,0),codebook(step,1)
        bits=(np.min(np.abs(values[:,None]-c1[None,:]),axis=1) <
              np.min(np.abs(values[:,None]-c0[None,:]),axis=1)).astype(np.uint8)
        return ExtractionResult(watermark_from_scrambled_bits(bits,n,self.config.arnold_iterations), None, {'fully_blind':True})
