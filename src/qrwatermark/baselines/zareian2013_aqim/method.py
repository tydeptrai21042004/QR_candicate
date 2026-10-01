from __future__ import annotations

from typing import Any

import numpy as np

from ...core.config import BaselineConfig
from ...core.interfaces import WatermarkMethod
from ...core.types import EmbeddingResult, ExtractionResult
from ..common import finish_payload, prepare_payload

_SQRT2 = float(np.sqrt(2.0))


def _haar_axis_forward(x: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    n = arr.shape[axis]
    if n % 2:
        raise ValueError("Haar transform requires an even axis length")
    even = np.take(arr, np.arange(0, n, 2), axis=axis)
    odd = np.take(arr, np.arange(1, n, 2), axis=axis)
    low = (even + odd) / _SQRT2
    high = (even - odd) / _SQRT2
    return np.concatenate([low, high], axis=axis)


def _haar_axis_inverse(x: np.ndarray, axis: int) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    n = arr.shape[axis]
    if n % 2:
        raise ValueError("Haar inverse requires an even axis length")
    half = n // 2
    low = np.take(arr, np.arange(0, half), axis=axis)
    high = np.take(arr, np.arange(half, n), axis=axis)
    even = (low + high) / _SQRT2
    odd = (low - high) / _SQRT2
    out = np.empty_like(arr)
    index_even = [slice(None)] * arr.ndim
    index_odd = [slice(None)] * arr.ndim
    index_even[axis] = slice(0, n, 2)
    index_odd[axis] = slice(1, n, 2)
    out[tuple(index_even)] = even
    out[tuple(index_odd)] = odd
    return out


def haar2_levels(block: np.ndarray, levels: int = 2) -> np.ndarray:
    """Orthonormal separable Haar transform with MATLAB/standard quadrant layout."""
    out = np.asarray(block, dtype=np.float64).copy()
    h, w = out.shape
    for level in range(levels):
        hh, ww = h // (2**level), w // (2**level)
        region = out[:hh, :ww]
        region = _haar_axis_forward(region, axis=1)
        region = _haar_axis_forward(region, axis=0)
        out[:hh, :ww] = region
    return out


def inverse_haar2_levels(coeffs: np.ndarray, levels: int = 2) -> np.ndarray:
    out = np.asarray(coeffs, dtype=np.float64).copy()
    h, w = out.shape
    for level in reversed(range(levels)):
        hh, ww = h // (2**level), w // (2**level)
        region = out[:hh, :ww]
        region = _haar_axis_inverse(region, axis=0)
        region = _haar_axis_inverse(region, axis=1)
        out[:hh, :ww] = region
    return out


def _block_entropy(block: np.ndarray) -> float:
    vals = np.clip(np.rint(block), 0, 255).astype(np.uint8).ravel()
    counts = np.bincount(vals, minlength=256).astype(np.float64)
    p = counts[counts > 0] / float(vals.size)
    return float(-np.sum(p * np.log2(p)))


def _nearest_integer_positive(x: float) -> float:
    # s and Delta are non-negative here. floor(x+1/2) gives deterministic
    # nearest-integer quantisation without NumPy's ties-to-even behaviour.
    return float(np.floor(float(x) + 0.5))


def adaptive_step(x: np.ndarray, delta0: float, gamma: float) -> float:
    """Zareian-Tohidypour Eq. (1)."""
    mean_abs = float(np.mean(np.abs(np.asarray(x, dtype=np.float64))))
    return float(delta0) * max(mean_abs, 1e-12) ** (1.0 / float(gamma))


def normalized_magnitude(x: np.ndarray) -> float:
    """Zareian-Tohidypour Eq. (2): RMS magnitude of the low-frequency vector."""
    arr = np.asarray(x, dtype=np.float64)
    return float(np.sqrt(np.mean(arr * arr)))


def qim_vector_level(s: float, bit: int, delta: float) -> float:
    """Zareian-Tohidypour Eq. (3), Q_b(s)."""
    b = int(bit)
    d = float(delta)
    k = _nearest_integer_positive((float(s) + b * d / 2.0) / d)
    return float(d * k - b * d / 2.0)


def _candidate_level_at_decoder(s_over_t: float, bit: int, delta_hat: float) -> float:
    # Paper Eq. (7), the same binary dithered QIM rule evaluated using Delta_hat.
    return qim_vector_level(float(s_over_t), int(bit), float(delta_hat))


class Zareian2013AdaptiveQIM(WatermarkMethod):
    """Zareian & Tohidypour, IET Image Processing 7(5), 432-441 (2013).

    Paper equations reproduced:
      * 16x16 non-overlapping blocks ranked by entropy;
      * two-level orthonormal Haar DWT;
      * one bit per selected block in the 4x4 level-2 LL vector;
      * adaptive power-law step, Eqs. (1)-(4);
      * decoder gain estimate and minimum-distance decision, Eqs. (5)-(8).

    The paper is a scalar-image method.  In this color-image repository it is
    applied to ``config.channel`` so that it can serve as a QIM control without
    introducing a color-space conversion unavailable to the other scalar
    baselines.  Its native capacity is at most one bit per 16x16 host block.
    """

    name = "zareian2013_aqim"
    strength_field = "quant_step"  # repository field used for paper Delta_0

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig(
            block_size=16,
            watermark_size=16,
            quant_step=0.21,
            gamma=3.25,
        )
        if self.config.block_size != 16:
            raise ValueError("Zareian2013 AQIM uses 16x16 image blocks")
        if self.config.quant_step <= 0.0:
            raise ValueError("quant_step stores paper Delta_0 and must be positive")
        if self.config.gamma <= 0.0:
            raise ValueError("gamma must be positive")

    @staticmethod
    def _all_positions(shape: tuple[int, int]) -> list[tuple[int, int]]:
        h, w = shape
        return [(r, c) for r in range(0, h - 15, 16) for c in range(0, w - 15, 16)]

    def _selected_positions(self, channel: np.ndarray, count: int) -> tuple[list[tuple[int, int]], np.ndarray]:
        positions = self._all_positions(channel.shape)
        if count > len(positions):
            raise ValueError(
                f"Zareian2013 capacity exceeded: payload={count} bits but a "
                f"{channel.shape[0]}x{channel.shape[1]} host has only {len(positions)} "
                "non-overlapping 16x16 carriers. Use a <=32x32 binary watermark "
                "for a 512x512 host, or the dedicated small-payload comparison config."
            )
        scores = [(_block_entropy(channel[r:r+16, c:c+16]), idx) for idx, (r, c) in enumerate(positions)]
        # Select the M highest-entropy blocks, then carry watermark bits in
        # raster order over that selected set.  This is recoverable from the
        # paper's one-bit-per-block position map alone; no hidden ordering
        # side channel is required.
        chosen = [idx for _, idx in sorted(scores, key=lambda z: (-z[0], z[1]))[:count]]
        chosen_set = set(chosen)
        selected = [p for idx, p in enumerate(positions) if idx in chosen_set]
        mask = np.zeros(len(positions), dtype=np.uint8)
        mask[chosen] = 1
        return selected, mask

    @staticmethod
    def _positions_from_mask(shape: tuple[int, int], mask: np.ndarray) -> list[tuple[int, int]]:
        positions = Zareian2013AdaptiveQIM._all_positions(shape)
        mask = np.asarray(mask, dtype=np.uint8).ravel()
        if mask.size != len(positions):
            raise ValueError("Zareian2013 block-position side information has the wrong length")
        return [p for p, keep in zip(positions, mask) if int(keep)]

    def embed(self, host: np.ndarray, watermark: np.ndarray, *, key: bytes) -> EmbeddingResult:
        del key  # the published selector is entropy-based and sends positions as side information
        c = self.config
        out = np.asarray(host, dtype=np.uint8).copy()
        channel = out[:, :, c.channel].astype(np.float64)
        bits = prepare_payload(watermark, c.arnold_iterations)
        selected, mask = self._selected_positions(channel, len(bits))
        all_positions = self._all_positions(channel.shape)
        means_y: list[float] = []

        for bit, (row, col) in zip(bits, selected):
            # The paper's reported Delta_0≈0.21 is on a normalized intensity
            # scale.  Work in [0,1] here, then return to 8-bit pixels after the
            # inverse transform; otherwise Delta_0 is 255x too small and is
            # erased by image quantisation.
            block = channel[row:row+16, col:col+16] / 255.0
            coeff = haar2_levels(block, levels=2)
            x = coeff[:4, :4].copy()
            delta = adaptive_step(x, float(c.quant_step), float(c.gamma))
            s = normalized_magnitude(x)
            sq = qim_vector_level(s, int(bit), delta)
            scale = 1.0 if s <= 1e-12 else sq / s
            y = scale * x
            coeff[:4, :4] = y
            marked = inverse_haar2_levels(coeff, levels=2)
            channel[row:row+16, col:col+16] = np.clip(np.rint(255.0 * marked), 0, 255)
            means_y.append(float(np.mean(y)))

        out[:, :, c.channel] = channel.astype(np.uint8)
        xi = float(np.sum(means_y))
        # Paper Sec. 4: one bit for every host block to signal selected blocks,
        # plus three nominal 8-bit words for Delta_0, gamma and xi.
        logical_side_bits = int(mask.size + 24)
        return EmbeddingResult(
            image=out,
            side_info={
                "selected_mask": mask,
                "delta0": float(c.quant_step),
                "gamma": float(c.gamma),
                "xi": xi,
            },
            metadata={
                "reference": "Zareian & Tohidypour 2013, IET Image Processing 7(5):432-441",
                "doi": "10.1049/iet-ipr.2013.0048",
                "carrier": "two-level Haar LL2 vector of high-entropy 16x16 blocks",
                "delta0": float(c.quant_step),
                "gamma": float(c.gamma),
                "side_information_bits": logical_side_bits,
                "native_capacity_bits": len(all_positions),
                "implementation_status": "published Eqs. (1)-(8); scalar method adapted to configured color channel",
            },
        )

    def extract(
        self,
        image: np.ndarray,
        *,
        key: bytes,
        side_info: dict[str, Any] | None = None,
        watermark_shape: tuple[int, int] = (16, 16),
    ) -> ExtractionResult:
        del key
        if side_info is None:
            raise ValueError("Zareian2013 AQIM requires the paper's decoder side information")
        required = {"selected_mask", "delta0", "gamma", "xi"}
        missing = required.difference(side_info)
        if missing:
            raise ValueError(f"Missing Zareian2013 side information: {sorted(missing)}")

        c = self.config
        channel = np.asarray(image, dtype=np.uint8)[:, :, c.channel].astype(np.float64)
        all_positions = self._all_positions(channel.shape)
        mask = np.asarray(side_info["selected_mask"], dtype=np.uint8).ravel()
        if mask.size != len(all_positions):
            raise ValueError("Zareian2013 block-position side information has the wrong length")
        count = int(watermark_shape[0] * watermark_shape[1])
        positions = self._positions_from_mask(channel.shape, mask)
        if len(positions) < count:
            raise ValueError("Insufficient Zareian2013 selected blocks in side information")
        positions = positions[:count]

        delta0 = float(side_info["delta0"])
        gamma = float(side_info["gamma"])
        xi = float(side_info["xi"])
        if abs(xi) <= 1e-12:
            raise ValueError("Zareian2013 gain side-information xi is numerically zero")

        z_vectors: list[np.ndarray] = []
        means_z: list[float] = []
        for row, col in positions:
            coeff = haar2_levels(channel[row:row+16, col:col+16] / 255.0, levels=2)
            z = coeff[:4, :4].copy()
            z_vectors.append(z)
            means_z.append(float(np.mean(z)))

        # Paper Eq. (5): global gain estimate T.
        gain = float(np.sum(means_z) / xi)
        if gain <= 1e-12:
            # The paper models a positive amplitude gain.  A non-positive
            # estimate falls outside that model and should not be silently
            # converted with abs(), which would alter Eqs. (5)-(7).
            gain = 1e-12

        bits = np.zeros(count, dtype=np.uint8)
        for i, z in enumerate(z_vectors):
            # Paper Eq. (6).
            delta_hat = delta0 * max(float(np.mean(np.abs(z))) / gain, 1e-12) ** (1.0 / gamma)
            s_prime = normalized_magnitude(z)
            normalized_received = s_prime / gain
            candidates = [
                _candidate_level_at_decoder(normalized_received, b, delta_hat)
                for b in (0, 1)
            ]
            # Paper Eq. (8), minimum Euclidean distance in this scalar level.
            bits[i] = int(np.argmin([abs(normalized_received - q) for q in candidates]))

        return ExtractionResult(
            watermark=finish_payload(bits, watermark_shape, c.arnold_iterations),
            metadata={"estimated_gain": gain},
        )
