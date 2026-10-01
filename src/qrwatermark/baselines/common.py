from __future__ import annotations

import numpy as np

from ..utils.permutation import selected_block_positions
from ..utils.watermark import arnold_transform, bits_from_watermark, inverse_arnold_transform, watermark_from_bits


def prepare_payload(watermark: np.ndarray, iterations: int) -> np.ndarray:
    return bits_from_watermark(arnold_transform(watermark, iterations))


def finish_payload(bits: np.ndarray, shape: tuple[int, int], iterations: int) -> np.ndarray:
    return inverse_arnold_transform(watermark_from_bits(bits, shape), iterations)


def keyed_positions(image_shape: tuple[int, int], block_size: int, count: int, key: bytes):
    return selected_block_positions(image_shape, block_size, count, key)


def canonical_qr(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Real QR with the positive-diagonal convention used by image QR papers.

    NumPy/LAPACK is free to flip signs of matching Q columns/R rows.  The
    watermark rules depend on individual Q/R entries, so we canonicalize every
    non-zero diagonal of R to be positive.
    """
    qmat, rmat = np.linalg.qr(np.asarray(block, dtype=np.float64))
    signs = np.sign(np.diag(rmat))
    signs[signs == 0.0] = 1.0
    d = np.diag(signs)
    return qmat @ d, d @ rmat


def qim_embed_phase(value: float, bit: int, q: float) -> float:
    """Quarter-period QIM used by Nha et al. (2022)/Sun-style embedding."""
    z = value % q
    if int(bit) == 0:
        return value + (q / 4.0 - z if z <= 3.0 * q / 4.0 else 5.0 * q / 4.0 - z)
    return value + (-q / 4.0 - z if z <= q / 4.0 else 3.0 * q / 4.0 - z)


def qim_decode_phase(value: float, q: float) -> int:
    return 0 if (value % q) <= q / 2.0 else 1


def relation_embed(a: float, b: float, bit: int, threshold: float) -> tuple[float, float]:
    """Relative modulation with a signed difference and a T-sized margin.

    This is the rule used by Chen et al. (2021) on quaternion Q coefficients
    and is also useful for paper-aligned QR relation baselines.
    """
    t = float(threshold)
    avg = 0.5 * (float(a) + float(b))
    if int(bit) == 1:
        if float(a) - float(b) < t:
            return avg + 0.5 * t, avg - 0.5 * t
    else:
        if float(a) - float(b) > -t:
            return avg - 0.5 * t, avg + 0.5 * t
    return float(a), float(b)
