from __future__ import annotations

import cv2
import numpy as np


def scaling_resample(image: np.ndarray, scale: float = 0.2) -> np.ndarray:
    h, w = image.shape[:2]
    small = cv2.resize(image, (max(1, int(round(w * scale))), max(1, int(round(h * scale)))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)


def registered_rotation_resample(image: np.ndarray, angle: float = 45.0) -> np.ndarray:
    """Rotate and inverse-rotate: a resampling attack, not unknown-rotation synchronization."""
    h, w = image.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    m1 = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    rot = cv2.warpAffine(image, m1, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    m2 = cv2.getRotationMatrix2D(center, -float(angle), 1.0)
    return cv2.warpAffine(rot, m2, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def rotation_unregistered(image: np.ndarray, angle: float = 5.0) -> np.ndarray:
    h, w = image.shape[:2]
    center = ((w - 1) / 2.0, (h - 1) / 2.0)
    m = cv2.getRotationMatrix2D(center, float(angle), 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def translation(image: np.ndarray, dx: int = 5, dy: int = 5) -> np.ndarray:
    h, w = image.shape[:2]
    m = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(image, m, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def crop_resize(image: np.ndarray, fraction: float = 0.1) -> np.ndarray:
    h, w = image.shape[:2]
    dy = int(round(h * fraction / 2)); dx = int(round(w * fraction / 2))
    crop = image[dy:h-dy, dx:w-dx]
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_CUBIC)


# Backward-compatible alias; this operation rotates and inverse-rotates and therefore
# measures registered resampling damage rather than synchronization robustness.
rotation_resample = registered_rotation_resample
