from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import hessenberg

from ...core.config import BaselineConfig
from ...core.interfaces import WatermarkMethod
from ...core.types import EmbeddingResult, ExtractionResult
from ..common import finish_payload, keyed_positions, prepare_payload


def _sgn(x: float) -> float:
    # MATLAB sign(0)=0, and the paper explicitly uses sign(.).
    return float(np.sign(float(x)))


def canonical_hessenberg(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hessenberg factorization with a deterministic column-sign convention.

    Hessenberg Q is not unique: for a diagonal D of +/-1 values,
    A = Q H Q^T = (Q D) (D H D) (Q D)^T.  SciPy/LAPACK and the MATLAB
    convention used in the paper can choose opposite Householder signs.  The
    published embedding equations multiply by sign(q_ij), so a stable
    convention is essential for a reproducible implementation.  We choose D
    so the available diagonal pivots of Q are non-negative while preserving A
    exactly.  Extraction itself uses absolute values, as in paper Eq. (14).
    """
    hmat, qmat = hessenberg(np.asarray(block, dtype=np.float64), calc_q=True)
    n = qmat.shape[0]
    signs = np.ones(n, dtype=np.float64)
    for j in range(1, n):
        pivot = float(qmat[j, j])
        if pivot < 0.0:
            signs[j] = -1.0
    d = np.diag(signs)
    return d @ hmat @ d, qmat @ d


class Su2016Hessenberg(WatermarkMethod):
    """Q. Su, IET Image Processing 10(11) (2016), 817-829.

    This corrects the repository's former approximation.  The paper embeds in
    q(2,2) and q(3,2) of the orthogonal Q from Hessenberg decomposition, not in
    an energy-selected coefficient of H.  Eqs. (11)-(14) are reproduced here.
    """

    name = "su2016_hessenberg"
    strength_field = "threshold"

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig(block_size=4, threshold=0.042)
        if self.config.block_size != 4:
            raise ValueError("Su2016 Hessenberg baseline is defined on 4x4 blocks")
        if self.config.threshold <= 0:
            raise ValueError("threshold T must be positive")

    @staticmethod
    def _modify_q(qmat: np.ndarray, bit: int, threshold: float) -> np.ndarray:
        q2 = qmat.copy()
        a = float(qmat[1, 1])  # paper q_{2,2}
        b = float(qmat[2, 1])  # paper q_{3,2}
        qavg = 0.5 * (a + b)
        half = 0.5 * float(threshold)
        if int(bit) == 1:  # paper Eq. (11)
            q2[1, 1] = _sgn(a) * (qavg + half)
            q2[2, 1] = _sgn(b) * (qavg - half)
        else:  # paper Eq. (12)
            q2[1, 1] = _sgn(a) * (qavg - half)
            q2[2, 1] = _sgn(b) * (qavg + half)
        return q2

    @staticmethod
    def _decode_q(qmat: np.ndarray) -> int:
        # Paper Eq. (14).
        return 1 if abs(float(qmat[1, 1])) >= abs(float(qmat[2, 1])) else 0

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
            hmat, qmat = canonical_hessenberg(block)
            q2 = self._modify_q(qmat, int(bit), float(c.threshold))
            # scipy.linalg.hessenberg returns A = Q H Q^T.
            candidate = q2 @ hmat @ q2.T
            channel[row : row + 4, col : col + 4] = np.clip(np.rint(candidate), 0, 255)

        out[:h0, :w0, c.channel] = channel[:h0, :w0].astype(np.uint8)
        return EmbeddingResult(
            image=out,
            side_info=None,
            metadata={
                "reference": "Q. Su 2016, IET Image Processing 10(11):817-829",
                "doi": "10.1049/iet-ipr.2016.0048",
                "carrier": "Q[1,1],Q[2,1]",
                "threshold": float(c.threshold),
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
            _, qmat = canonical_hessenberg(block)
            bits[i] = self._decode_q(qmat)

        return ExtractionResult(watermark=finish_payload(bits, watermark_shape, c.arnold_iterations))
