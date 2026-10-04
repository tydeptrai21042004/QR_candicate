from __future__ import annotations

from typing import Any

import numpy as np
from scipy.linalg import schur

from ...core.config import BaselineConfig
from ...core.interfaces import WatermarkMethod
from ...core.types import EmbeddingResult, ExtractionResult
from ..common import finish_payload, keyed_positions, prepare_payload

MODE_U = 0
MODE_D = 1


def _sgn_nonzero(x: float) -> float:
    # The published equations use sign(.).  A zero entry is a measure-zero
    # event for natural-image Schur vectors; use +1 to avoid erasing a carrier.
    return -1.0 if float(x) < 0.0 else 1.0


def canonical_schur(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Real Schur A = U D U^T with deterministic column signs.

    Real Schur vectors are sign-ambiguous. Multiplying U by a diagonal +/-1
    matrix and applying the same similarity to D leaves A unchanged. The paper
    decodes a signed relation between rows 2 and 3, so we choose an equivalent
    deterministic carrier-pair sign before applying that relation.
    """
    dmat, umat = schur(np.asarray(block, dtype=np.float64), output="real")
    n = umat.shape[1]
    signs = np.ones(n, dtype=np.float64)
    for j in range(n):
        # Eq. (16) compares the *signed* U(2,c), U(3,c), while Eqs. (7)-(8)
        # embed by their magnitudes.  Schur vectors have an arbitrary ± sign,
        # so the paper's detector is not numerically well-defined until that
        # ambiguity is fixed.  For the carrier pair, choose the equivalent
        # column sign that makes rows 2 and 3 jointly non-positive whenever
        # they share a sign (the natural-image case used by the paper).
        pair_sum = float(umat[1, j] + umat[2, j])
        if pair_sum > 0.0:
            signs[j] = -1.0
        elif abs(pair_sum) <= 1e-15:
            k = int(np.argmax(np.abs(umat[:, j])))
            if float(umat[k, j]) > 0.0:
                signs[j] = -1.0
    s = np.diag(signs)
    return s @ dmat @ s, umat @ s


def _dominant_eigen_column(dmat: np.ndarray) -> int:
    """Column c associated with the maximum (real) Schur eigenvalue.

    The paper calls this the column of D_max.  For a real triangular Schur
    block the eigenvalues are its diagonal entries; using the largest diagonal
    entry is the direct real-valued interpretation of Eqs. (5)-(6)/(14)-(15).
    """
    diagonal = np.real(np.diag(np.asarray(dmat, dtype=np.float64)))
    return int(np.argmax(diagonal))


def _u_candidate(umat: np.ndarray, dmat: np.ndarray, c: int, bit: int, threshold: float) -> np.ndarray:
    """Paper Eqs. (7)-(9): embed in u_{2,c}, u_{3,c}."""
    u2 = umat.copy()
    a = float(umat[1, c])
    b = float(umat[2, c])
    avg = 0.5 * (abs(a) + abs(b))
    half = 0.5 * float(threshold)
    if int(bit) == 1:
        u2[1, c] = _sgn_nonzero(a) * (avg + half)
        u2[2, c] = _sgn_nonzero(b) * (avg - half)
    else:
        u2[1, c] = _sgn_nonzero(a) * (avg - half)
        u2[2, c] = _sgn_nonzero(b) * (avg + half)
    return u2 @ dmat @ u2.T


def _d_embed(value: float, bit: int, delta: float) -> float:
    """Paper Eq. (10): quarter/three-quarter quantization of D_max."""
    delta = float(delta)
    base = float(value) - np.mod(float(value), delta)
    return base + (0.75 * delta if int(bit) == 1 else 0.25 * delta)


def _d_decode(value: float, delta: float) -> int:
    """Paper Eq. (17)."""
    return 0 if np.mod(float(value), float(delta)) < 0.5 * float(delta) else 1


def _u_decode_paper(umat: np.ndarray, c: int) -> int:
    """Paper Eq. (16): raw relation after selecting the D_max column.

    The printed rule is 0 when u_{2,c} > u_{3,c}, otherwise 1.  We keep that
    rule rather than replacing it with an easier absolute-value surrogate.
    """
    return 0 if float(umat[1, c]) > float(umat[2, c]) else 1


class Su2020Schur(WatermarkMethod):
    """Su, Zhang & Wang, Soft Computing 24 (2020), 445-460.

    This implementation follows the published two-candidate method:
      1) embed the bit into U using Eqs. (7)-(9);
      2) embed the same bit into D_max using Eqs. (10)-(11);
      3) retain the candidate with smaller squared pixel distortion, Eq. (12);
      4) retain one mode flag per embedded bit, Eq. (13);
      5) decode U/D using Eqs. (16)/(17).

    The repository keeps its common binary payload, Arnold transform and keyed
    block ordering so capacity/attacks are standardized across baselines.
    """

    name = "su2020_schur"
    # Both published parameters jointly determine the selected candidate.
    strength_fields = ("threshold", "quant_step")

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig(block_size=4, threshold=0.03, quant_step=25.0)
        if self.config.block_size != 4:
            raise ValueError("Su2020 Schur baseline is defined on 4x4 blocks")
        if self.config.threshold <= 0.0 or self.config.quant_step <= 0.0:
            raise ValueError("Su2020 requires positive T and Delta")

    def embed(self, host: np.ndarray, watermark: np.ndarray, *, key: bytes) -> EmbeddingResult:
        c = self.config
        out = np.asarray(host, dtype=np.uint8).copy()
        bits = prepare_payload(watermark, c.arnold_iterations)
        channel = out[:, :, c.channel].astype(np.float64)
        h0 = (channel.shape[0] // 4) * 4
        w0 = (channel.shape[1] // 4) * 4
        positions = keyed_positions((h0, w0), 4, len(bits), key)
        flags = np.zeros(len(bits), dtype=np.uint8)

        for i, (bit, (row, col)) in enumerate(zip(bits, positions)):
            block = channel[row : row + 4, col : col + 4]
            dmat, umat = canonical_schur(block)
            col_idx = _dominant_eigen_column(dmat)

            # Method 1: U candidate, paper Eqs. (7)-(9).
            cand_u = _u_candidate(umat, dmat, col_idx, int(bit), float(c.threshold))

            # Method 2: D candidate, paper Eqs. (10)-(11).
            d2 = dmat.copy()
            d2[col_idx, col_idx] = _d_embed(
                float(dmat[col_idx, col_idx]), int(bit), float(c.quant_step)
            )
            cand_d = umat @ d2 @ umat.T

            # Eq. (12): choose the lower total squared pixel change.
            err_u = float(np.sum((block - cand_u) ** 2))
            err_d = float(np.sum((block - cand_d) ** 2))
            if err_u < err_d:
                candidate = cand_u
                flags[i] = MODE_U
            else:
                candidate = cand_d
                flags[i] = MODE_D

            channel[row : row + 4, col : col + 4] = np.clip(np.rint(candidate), 0, 255)

        out[:h0, :w0, c.channel] = channel[:h0, :w0].astype(np.uint8)
        return EmbeddingResult(
            image=out,
            side_info={"flags": flags},
            metadata={
                "reference": "Su, Zhang & Wang 2020, Soft Computing 24:445-460",
                "doi": "10.1007/s00500-019-03924-5",
                "carrier": "lower-distortion of Schur U relation or D_max QIM",
                "threshold": float(c.threshold),
                "quant_step": float(c.quant_step),
                "side_information_bits": int(flags.size),
                "implementation_status": "published Eqs. (5)-(17), standardized binary protocol",
                "information_model": "side-information-assisted/semi-blind: published mode flags required",
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
        if side_info is None or "flags" not in side_info:
            raise ValueError("Su2020 Schur baseline requires its published embedding-mode flags")
        c = self.config
        flags = np.asarray(side_info["flags"], dtype=np.uint8).ravel()
        count = int(watermark_shape[0] * watermark_shape[1])
        if flags.size < count:
            raise ValueError("Insufficient Su2020 Schur mode flags")

        channel = np.asarray(image, dtype=np.uint8)[:, :, c.channel].astype(np.float64)
        h0 = (channel.shape[0] // 4) * 4
        w0 = (channel.shape[1] // 4) * 4
        positions = keyed_positions((h0, w0), 4, count, key)
        bits = np.zeros(count, dtype=np.uint8)

        for i, (row, col) in enumerate(positions):
            block = channel[row : row + 4, col : col + 4]
            dmat, umat = canonical_schur(block)
            col_idx = _dominant_eigen_column(dmat)
            if int(flags[i]) == MODE_U:
                bits[i] = _u_decode_paper(umat, col_idx)
            else:
                bits[i] = _d_decode(float(dmat[col_idx, col_idx]), float(c.quant_step))

        return ExtractionResult(watermark=finish_payload(bits, watermark_shape, c.arnold_iterations))
