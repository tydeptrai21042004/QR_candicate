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


def qim_embed_phase(value: float, bit: int, q: float) -> float:
    z = value % q
    if int(bit) == 0:
        return value + (q / 4.0 - z if z <= 3.0 * q / 4.0 else 5.0 * q / 4.0 - z)
    return value + (-q / 4.0 - z if z <= q / 4.0 else 3.0 * q / 4.0 - z)


def qim_decode_phase(value: float, q: float) -> int:
    return 0 if (value % q) <= q / 2.0 else 1
