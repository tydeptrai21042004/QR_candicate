from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np


def prepare_binary_watermark(source: str | Path | np.ndarray, size: int = 64, background: int = 255) -> np.ndarray:
    if isinstance(source, (str, Path)):
        img = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise FileNotFoundError(f"Cannot read watermark: {source}")
    else:
        img = np.asarray(source)

    if img.ndim == 3 and img.shape[2] == 4:
        bgr = img[..., :3].astype(np.float32)
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        bg = np.full_like(bgr, float(background), dtype=np.float32)
        gray = cv2.cvtColor((bgr * alpha + bg * (1.0 - alpha)).astype(np.uint8), cv2.COLOR_BGR2GRAY)
    elif img.ndim == 3:
        gray = cv2.cvtColor(img[..., :3].astype(np.uint8), cv2.COLOR_BGR2GRAY)
    else:
        gray = img.astype(np.uint8)

    if gray.shape != (size, size):
        gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    return ((gray >= 128).astype(np.uint8) * 255)


def bits_from_watermark(wm: np.ndarray) -> np.ndarray:
    if wm.ndim == 3:
        wm = cv2.cvtColor(wm, cv2.COLOR_BGR2GRAY)
    return (wm >= 128).astype(np.uint8).ravel()


def watermark_from_bits(bits: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(bits, dtype=np.uint8).reshape(shape)
    return arr * 255


@lru_cache(maxsize=32)
def _arnold_destination_map(n: int, iterations: int, inverse: bool) -> np.ndarray:
    """Return the exact final destination index of each source pixel.

    The old implementation performed ``iterations * n^2`` Python-loop writes.
    The Arnold map is geometry-only, so a session can precompute the final
    permutation once and apply it with one NumPy indexed assignment per frame.
    This is bit-for-bit identical to the reference loops.
    """
    n = int(n)
    it = int(iterations)
    if n <= 0:
        raise ValueError("Arnold size must be positive")
    if it < 0:
        raise ValueError("Arnold iterations must be non-negative")

    x, y = np.indices((n, n), dtype=np.int64)
    x = x.ravel()
    y = y.ravel()
    for _ in range(it):
        if inverse:
            x, y = (2 * x - y) % n, (-x + y) % n
        else:
            x, y = (x + y) % n, (x + 2 * y) % n
    dst = (x * n + y).astype(np.intp, copy=False)
    dst.flags.writeable = False
    return dst


def arnold_transform(image: np.ndarray, iterations: int) -> np.ndarray:
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Arnold transform requires a square 2D watermark")
    src = np.asarray(image)
    if int(iterations) == 0:
        return src.copy()
    dst = _arnold_destination_map(src.shape[0], int(iterations), False)
    out = np.empty_like(src)
    out.ravel()[dst] = src.ravel()
    return out


def inverse_arnold_transform(image: np.ndarray, iterations: int) -> np.ndarray:
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Arnold transform requires a square 2D watermark")
    src = np.asarray(image)
    if int(iterations) == 0:
        return src.copy()
    dst = _arnold_destination_map(src.shape[0], int(iterations), True)
    out = np.empty_like(src)
    out.ravel()[dst] = src.ravel()
    return out


def arnold_destination_map(size: int, iterations: int, inverse: bool = False) -> np.ndarray:
    """Public read-only Arnold destination LUT for software/FPGA control planes.

    The returned vector maps every source flat index to its final destination
    index after ``iterations`` Arnold steps.  It is geometry-only and can be
    stored once in ROM/BRAM for a streaming hardware implementation.
    """
    return _arnold_destination_map(int(size), int(iterations), bool(inverse))


def scrambled_bits_from_watermark(wm: np.ndarray, iterations: int) -> np.ndarray:
    """Return exactly the bits produced by ``bits_from_watermark(arnold_transform(...))``.

    This avoids allocating and permuting an intermediate 2-D uint8 image in the
    online path.  It is bit-for-bit equivalent to the original operation.
    """
    arr = np.asarray(wm)
    if arr.ndim != 2 or arr.shape[0] != arr.shape[1]:
        raise ValueError("Arnold transform requires a square 2D watermark")
    src_bits = (arr >= 128).astype(np.uint8, copy=False).ravel()
    if int(iterations) == 0:
        return src_bits.copy()
    dst = _arnold_destination_map(arr.shape[0], int(iterations), False)
    out = np.empty_like(src_bits)
    out[dst] = src_bits
    return out


def watermark_from_scrambled_bits(bits: np.ndarray, size: int, iterations: int) -> np.ndarray:
    """Inverse Arnold mapping directly in the binary-bit domain.

    Equivalent to ``inverse_arnold_transform(watermark_from_bits(bits), ...)``
    but avoids an extra image-domain permutation in the decoder hot path.
    """
    n = int(size)
    src = np.asarray(bits, dtype=np.uint8).ravel()
    if src.size != n * n:
        raise ValueError(f"expected {n*n} bits, got {src.size}")
    if int(iterations) == 0:
        return (src.reshape(n, n) * 255).astype(np.uint8, copy=False)
    dst = _arnold_destination_map(n, int(iterations), True)
    out = np.empty_like(src)
    out[dst] = src
    return (out.reshape(n, n) * 255).astype(np.uint8, copy=False)
