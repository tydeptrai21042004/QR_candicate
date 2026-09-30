from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import schur

from ...core.config import BaselineConfig
from ...core.interfaces import WatermarkMethod
from ...core.types import EmbeddingResult, ExtractionResult
from ..common import finish_payload, keyed_positions, prepare_payload, qim_decode_phase, qim_embed_phase

MODE_T = 0
MODE_U = 1


def _nearest_axis_angle(phi: float, bit: int) -> float:
    # bit 0 -> first coordinate dominant; bit 1 -> second coordinate dominant.
    base = 0.0 if int(bit) == 0 else 0.5 * np.pi
    k = round((phi - base) / np.pi)
    return base + k * np.pi


def _unitary_candidate(z: np.ndarray, t: np.ndarray, k: int, bit: int) -> np.ndarray:
    if z.shape[0] < 3:
        raise ValueError("Schur baseline requires >=3x3 blocks")
    a = float(z[1, k]); b = float(z[2, k])
    phi = float(np.arctan2(b, a))
    target = _nearest_axis_angle(phi, bit)
    theta = target - phi
    c, s = float(np.cos(theta)), float(np.sin(theta))
    g = np.eye(z.shape[0], dtype=np.float64)
    g[1, 1] = c; g[1, 2] = -s; g[2, 1] = s; g[2, 2] = c
    z2 = g @ z
    return z2 @ t @ z2.T


def _unitary_decode(z: np.ndarray, k: int) -> int:
    return 0 if abs(float(z[1, k])) >= abs(float(z[2, k])) else 1


class Su2020Schur(WatermarkMethod):
    """Transparent reconstruction of the two-candidate Schur scheme in Su et al. (2020).

    Public descriptions state that one candidate embeds in the Schur triangular
    matrix, another in the unitary matrix, and the lower-distortion candidate is
    selected while a mode flag is retained. This implementation reproduces that
    structure under the unified binary benchmark; it is not the authors' source.
    """

    name = "su2020_schur"

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig(block_size=4, quant_step=8.0)

    @staticmethod
    def _k(t: np.ndarray) -> int:
        # Dominant Schur diagonal index, used as a stable energy anchor.
        return int(np.argmax(np.abs(np.diag(t))))

    def embed(self, host: np.ndarray, watermark: np.ndarray, *, key: bytes) -> EmbeddingResult:
        c = self.config
        out = np.asarray(host, dtype=np.uint8).copy()
        bits = prepare_payload(watermark, c.arnold_iterations)
        channel = out[:, :, c.channel].astype(np.float64)
        h0 = (channel.shape[0] // c.block_size) * c.block_size
        w0 = (channel.shape[1] // c.block_size) * c.block_size
        pos = keyed_positions((h0, w0), c.block_size, len(bits), key)
        flags = np.zeros(len(bits), dtype=np.uint8)
        for i, (bit, (row, col)) in enumerate(zip(bits, pos)):
            block = channel[row : row + c.block_size, col : col + c.block_size]
            t, z = schur(block, output="real")
            k = self._k(t)

            t2 = t.copy()
            sign = 1.0 if float(t2[k, k]) >= 0 else -1.0
            t2[k, k] = sign * qim_embed_phase(abs(float(t2[k, k])), int(bit), c.quant_step)
            cand_t = z @ t2 @ z.T
            cand_u = _unitary_candidate(z, t, k, int(bit))
            mse_t = float(np.mean((block - cand_t) ** 2))
            mse_u = float(np.mean((block - cand_u) ** 2))
            if mse_t <= mse_u:
                cand, flags[i] = cand_t, MODE_T
            else:
                cand, flags[i] = cand_u, MODE_U
            channel[row : row + c.block_size, col : col + c.block_size] = np.clip(np.rint(cand), 0, 255)
        out[:h0, :w0, c.channel] = channel[:h0, :w0].astype(np.uint8)
        return EmbeddingResult(image=out, side_info={"flags": flags}, metadata={"reference": "Su et al. 2020", "side_information_bits": int(flags.size)})

    def extract(self, image: np.ndarray, *, key: bytes, side_info: dict[str, Any] | None = None, watermark_shape=(64, 64)) -> ExtractionResult:
        if side_info is None or "flags" not in side_info:
            raise ValueError("Su2020 Schur baseline requires its embedding-mode flags")
        c = self.config
        flags = np.asarray(side_info["flags"], dtype=np.uint8).ravel()
        l = int(watermark_shape[0] * watermark_shape[1])
        if len(flags) < l:
            raise ValueError("Insufficient Schur mode flags")
        channel = np.asarray(image, dtype=np.uint8)[:, :, c.channel].astype(np.float64)
        h0 = (channel.shape[0] // c.block_size) * c.block_size
        w0 = (channel.shape[1] // c.block_size) * c.block_size
        pos = keyed_positions((h0, w0), c.block_size, l, key)
        bits = np.zeros(l, dtype=np.uint8)
        for i, (row, col) in enumerate(pos):
            block = channel[row : row + c.block_size, col : col + c.block_size]
            t, z = schur(block, output="real")
            k = self._k(t)
            if int(flags[i]) == MODE_T:
                bits[i] = qim_decode_phase(abs(float(t[k, k])), c.quant_step)
            else:
                bits[i] = _unitary_decode(z, k)
        return ExtractionResult(watermark=finish_payload(bits, watermark_shape, c.arnold_iterations))
