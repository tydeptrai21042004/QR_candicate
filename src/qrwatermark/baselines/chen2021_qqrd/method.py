from __future__ import annotations

from typing import Any

import numpy as np

from ...core.config import BaselineConfig
from ...core.interfaces import WatermarkMethod
from ...core.types import EmbeddingResult, ExtractionResult
from ..common import finish_payload, keyed_positions, prepare_payload, relation_embed


def qconj(q: np.ndarray) -> np.ndarray:
    out = np.asarray(q, dtype=np.float64).copy()
    out[..., 1:] *= -1.0
    return out


def qmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product, vectorized over all leading dimensions."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    a0, a1, a2, a3 = np.moveaxis(a, -1, 0)
    b0, b1, b2, b3 = np.moveaxis(b, -1, 0)
    return np.stack(
        [
            a0 * b0 - a1 * b1 - a2 * b2 - a3 * b3,
            a0 * b1 + a1 * b0 + a2 * b3 - a3 * b2,
            a0 * b2 - a1 * b3 + a2 * b0 + a3 * b1,
            a0 * b3 + a1 * b2 - a2 * b1 + a3 * b0,
        ],
        axis=-1,
    )


def qinner(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.sum(qmul(qconj(x), y), axis=0)


def qnorm(x: np.ndarray) -> float:
    z = qinner(x, x)
    return float(np.sqrt(max(float(z[0]), 0.0)))


def quaternion_qr(a: np.ndarray, eps: float = 1e-10) -> tuple[np.ndarray, np.ndarray]:
    """Quaternion modified Gram-Schmidt QR with positive real R diagonal.

    Chen et al. accelerate the same quaternion QR factorization through a real
    structure-preserving algorithm.  This compact reference implementation
    computes the quaternion factorization directly, preserving the mathematical
    Q/R carrier while remaining independent of unavailable author source code.
    """
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 3 or a.shape[0] != a.shape[1] or a.shape[2] != 4:
        raise ValueError("quaternion_qr expects an (n,n,4) quaternion matrix")
    n = a.shape[0]
    q = np.zeros_like(a)
    r = np.zeros((n, n, 4), dtype=np.float64)

    for j in range(n):
        v = a[:, j, :].copy()
        for i in range(j):
            rij = qinner(q[:, i, :], v)
            r[i, j, :] = rij
            v -= qmul(q[:, i, :], np.broadcast_to(rij, q[:, i, :].shape))
        nrm = qnorm(v)
        diagonal = nrm
        if nrm <= eps:
            # For a rank-deficient column the true R[j,j] is zero. Complete Q
            # deterministically for a valid orthonormal basis, but keep that
            # zero diagonal so Q@R still reconstructs the original matrix.
            found = False
            for basis_idx in range(n):
                v2 = np.zeros((n, 4), dtype=np.float64)
                v2[basis_idx, 0] = 1.0
                for i in range(j):
                    proj = qinner(q[:, i, :], v2)
                    v2 -= qmul(q[:, i, :], np.broadcast_to(proj, v2.shape))
                n2 = qnorm(v2)
                if n2 > eps:
                    v = v2
                    nrm = n2
                    found = True
                    break
            if not found:
                raise np.linalg.LinAlgError("Unable to complete quaternion QR basis")
        q[:, j, :] = v / nrm
        r[j, j, 0] = diagonal
    return q, r


def quaternion_matmul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n, k = a.shape[:2]
    k2, m = b.shape[:2]
    if k != k2:
        raise ValueError("Quaternion matrix dimensions do not align")
    out = np.zeros((n, m, 4), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            acc = np.zeros(4, dtype=np.float64)
            for p in range(k):
                acc += qmul(a[i, p, :], b[p, j, :])
            out[i, j, :] = acc
    return out


def bgr_to_pure_quaternion(block: np.ndarray) -> np.ndarray:
    """OpenCV BGR -> q = R*i + G*j + B*k."""
    block = np.asarray(block, dtype=np.float64)
    out = np.zeros(block.shape[:2] + (4,), dtype=np.float64)
    out[..., 1] = block[..., 2]
    out[..., 2] = block[..., 1]
    out[..., 3] = block[..., 0]
    return out


def pure_quaternion_to_bgr(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return np.stack([q[..., 3], q[..., 2], q[..., 1]], axis=-1)


class Chen2021QuaternionQR(WatermarkMethod):
    """Chen et al., Signal Processing 185 (2021), 108088.

    Each 4x4 RGB block is a pure quaternion matrix. Three bits are embedded per
    block by relative modulation of q21/q31 on the i, j and k imaginary parts,
    exactly following Eqs. (4)-(6) of the paper.
    """

    name = "chen2021_qqrd"
    strength_field = "threshold"

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig(block_size=4, threshold=0.03)
        if self.config.block_size != 4:
            raise ValueError("Chen2021 QQRD baseline is defined on 4x4 quaternion blocks")
        if self.config.threshold <= 0:
            raise ValueError("threshold T must be positive")

    def embed(self, host: np.ndarray, watermark: np.ndarray, *, key: bytes) -> EmbeddingResult:
        c = self.config
        out = np.asarray(host, dtype=np.uint8).copy()
        if out.ndim != 3 or out.shape[2] != 3:
            raise ValueError("Chen2021 QQRD requires a 3-channel color host image")
        bits = prepare_payload(watermark, c.arnold_iterations)
        h0 = (out.shape[0] // 4) * 4
        w0 = (out.shape[1] // 4) * 4
        block_count = int((len(bits) + 2) // 3)
        positions = keyed_positions((h0, w0), 4, block_count, key)

        bit_index = 0
        for row, col in positions:
            block = out[row : row + 4, col : col + 4, :].astype(np.float64)
            aq = bgr_to_pure_quaternion(block)
            qmat, rmat = quaternion_qr(aq)
            q2 = qmat.copy()
            for imag in (1, 2, 3):
                if bit_index >= len(bits):
                    break
                a = float(q2[1, 0, imag])  # q21, selected imaginary part
                b = float(q2[2, 0, imag])  # q31
                a2, b2 = relation_embed(a, b, int(bits[bit_index]), float(c.threshold))
                q2[1, 0, imag] = a2
                q2[2, 0, imag] = b2
                bit_index += 1
            candidate_q = quaternion_matmul(q2, rmat)
            candidate = pure_quaternion_to_bgr(candidate_q)
            out[row : row + 4, col : col + 4, :] = np.clip(np.rint(candidate), 0, 255).astype(np.uint8)

        return EmbeddingResult(
            image=out,
            side_info=None,
            metadata={
                "reference": "Chen et al. 2021, Signal Processing 185:108088",
                "doi": "10.1016/j.sigpro.2021.108088",
                "carrier": "imaginary parts of quaternion Q[1,0],Q[2,0]",
                "bits_per_block": 3,
                "threshold": float(c.threshold),
                "factorization": "direct quaternion QR reference implementation",
                "side_information_bits": 0,
                "information_model": "blind",
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
        arr = np.asarray(image, dtype=np.uint8)
        if arr.ndim != 3 or arr.shape[2] != 3:
            raise ValueError("Chen2021 QQRD requires a 3-channel color image")
        h0 = (arr.shape[0] // 4) * 4
        w0 = (arr.shape[1] // 4) * 4
        count = int(watermark_shape[0] * watermark_shape[1])
        block_count = int((count + 2) // 3)
        positions = keyed_positions((h0, w0), 4, block_count, key)
        bits = np.zeros(count, dtype=np.uint8)

        bit_index = 0
        for row, col in positions:
            block = arr[row : row + 4, col : col + 4, :].astype(np.float64)
            qmat, _ = quaternion_qr(bgr_to_pure_quaternion(block))
            for imag in (1, 2, 3):
                if bit_index >= count:
                    break
                # Paper Eq. (6), same imaginary unit on both coefficients.
                bits[bit_index] = 1 if float(qmat[1, 0, imag]) >= float(qmat[2, 0, imag]) else 0
                bit_index += 1

        return ExtractionResult(watermark=finish_payload(bits, watermark_shape, c.arnold_iterations))
