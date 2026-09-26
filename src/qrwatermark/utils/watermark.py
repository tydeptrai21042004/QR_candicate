from __future__ import annotations

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


def arnold_transform(image: np.ndarray, iterations: int) -> np.ndarray:
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Arnold transform requires a square 2D watermark")
    n = image.shape[0]
    result = image.copy()
    for _ in range(iterations):
        new = np.empty_like(result)
        for x in range(n):
            for y in range(n):
                new[(x + y) % n, (x + 2 * y) % n] = result[x, y]
        result = new
    return result


def inverse_arnold_transform(image: np.ndarray, iterations: int) -> np.ndarray:
    if image.ndim != 2 or image.shape[0] != image.shape[1]:
        raise ValueError("Arnold transform requires a square 2D watermark")
    n = image.shape[0]
    result = image.copy()
    for _ in range(iterations):
        new = np.empty_like(result)
        for x in range(n):
            for y in range(n):
                new[(2 * x - y) % n, (-x + y) % n] = result[x, y]
        result = new
    return result
