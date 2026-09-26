from __future__ import annotations

import numpy as np


def gaussian_noise(image: np.ndarray, mean: float = 0.0, variance: float = 0.003, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = image.astype(np.float32) / 255.0
    y = x + rng.normal(mean, np.sqrt(variance), size=x.shape).astype(np.float32)
    return np.clip(np.rint(y * 255.0), 0, 255).astype(np.uint8)


def salt_pepper_noise(image: np.ndarray, density: float = 0.1, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = image.copy()
    mask = rng.random(image.shape[:2])
    pepper = mask < density / 2.0
    salt = (mask >= density / 2.0) & (mask < density)
    out[pepper] = 0
    out[salt] = 255
    return out


def speckle_noise(image: np.ndarray, variance: float = 0.01, seed: int | None = None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = image.astype(np.float32) / 255.0
    n = rng.normal(0.0, np.sqrt(variance), size=x.shape).astype(np.float32)
    return np.clip(np.rint((x + x * n) * 255.0), 0, 255).astype(np.uint8)
