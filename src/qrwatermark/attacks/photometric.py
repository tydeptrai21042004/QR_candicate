from __future__ import annotations

import cv2
import numpy as np


def gamma_correction(image: np.ndarray, gamma: float = 1.5) -> np.ndarray:
    x = image.astype(np.float32) / 255.0
    return np.clip(np.rint((x ** float(gamma)) * 255.0), 0, 255).astype(np.uint8)


def brightness_contrast(image: np.ndarray, alpha: float = 1.1, beta: float = 10.0) -> np.ndarray:
    return np.clip(np.rint(image.astype(np.float32) * float(alpha) + float(beta)), 0, 255).astype(np.uint8)


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    ycc = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    ycc[:, :, 0] = cv2.equalizeHist(ycc[:, :, 0])
    return cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)


def clahe(image: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    cl = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(8, 8))
    lab[:, :, 0] = cl.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
