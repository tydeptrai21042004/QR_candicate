from __future__ import annotations

import math

import cv2
import numpy as np
from skimage.metrics import structural_similarity


def _gray_bits(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr.astype(np.uint8), cv2.COLOR_BGR2GRAY)
    return (arr.astype(np.uint8) >= 128).astype(np.uint8)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"PSNR requires equal shapes: {a.shape} vs {b.shape}")
    x = a.astype(np.float64); y = b.astype(np.float64)
    mse = float(np.mean((x - y) ** 2))
    return float("inf") if mse <= 1e-15 else 10.0 * math.log10(255.0 ** 2 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        raise ValueError(f"SSIM requires equal shapes: {a.shape} vs {b.shape}")
    if a.ndim == 2:
        return float(structural_similarity(a, b, data_range=255))
    return float(structural_similarity(a, b, data_range=255, channel_axis=-1))


def nc(original: np.ndarray, extracted: np.ndarray) -> float:
    """Normalized correlation with no ground-truth-driven inversion or alignment."""
    a = _gray_bits(original).astype(np.float64)
    b = _gray_bits(extracted).astype(np.float64)
    if a.shape != b.shape:
        raise ValueError(f"NC requires equal watermark shapes: {a.shape} vs {b.shape}")
    a = a.ravel(); b = b.ravel()
    den = float(np.sqrt(np.sum(a * a) * np.sum(b * b)))
    if den <= 1e-15:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.sum(a * b) / den)


def ber(original: np.ndarray, extracted: np.ndarray) -> float:
    a = _gray_bits(original); b = _gray_bits(extracted)
    if a.shape != b.shape:
        raise ValueError(f"BER requires equal watermark shapes: {a.shape} vs {b.shape}")
    return float(np.mean(a != b))


def bit_accuracy(original: np.ndarray, extracted: np.ndarray) -> float:
    return 1.0 - ber(original, extracted)
