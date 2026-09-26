from __future__ import annotations

import numpy as np


def random_occlusion(image: np.ndarray, fraction: float = 0.5, seed: int | None = 3, value: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = image.copy()
    mask = rng.random(image.shape[:2]) < float(fraction)
    out[mask] = int(value)
    return out
