from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def read_color(path: str | Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def write_image(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr=np.asarray(image)
    if arr.dtype!=np.uint8:
        arr=np.clip(np.rint(arr),0,255).astype(np.uint8)
    ok = cv2.imwrite(str(path), arr)
    if not ok:
        raise IOError(f"Cannot write image: {path}")
