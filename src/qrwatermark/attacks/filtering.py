from __future__ import annotations

import cv2
import numpy as np


def gaussian_blur(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    return cv2.GaussianBlur(image, (0, 0), float(sigma), borderType=cv2.BORDER_REFLECT_101)


def sharpen(image: np.ndarray, sigma: float = 1.0, amount: float = 1.5) -> np.ndarray:
    blur = cv2.GaussianBlur(image, (0, 0), float(sigma), borderType=cv2.BORDER_REFLECT_101)
    out = image.astype(np.float32) + float(amount) * (image.astype(np.float32) - blur.astype(np.float32))
    return np.clip(np.rint(out), 0, 255).astype(np.uint8)


def lowpass_filter(image: np.ndarray, kx: int = 9, ky: int = 9) -> np.ndarray:
    kx = int(kx) | 1; ky = int(ky) | 1
    return cv2.GaussianBlur(image, (kx, ky), 0)


def median_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.medianBlur(image, int(ksize) | 1)


def average_filter(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    return cv2.blur(image, (int(ksize), int(ksize)))


def motion_blur(image: np.ndarray, ksize: int = 7) -> np.ndarray:
    ksize = int(ksize) | 1
    kernel = np.zeros((ksize, ksize), dtype=np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    return cv2.filter2D(image, -1, kernel)
