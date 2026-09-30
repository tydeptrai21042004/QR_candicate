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


def certified_convex_gaussian(image: np.ndarray, sigma0: float = 0.50, sigma1: float = 0.75, alpha: float = 0.0, ksize: int = 3) -> np.ndarray:
    """Real-valued reflect-101 convolution from the proposal's certified hull.

    alpha=0 gives sigma0, alpha=1 gives sigma1, and intermediate values use
    the convex kernel mixture.  No output requantization is applied because the
    theorem is stated for the real-valued convolution operator.
    """
    if not 0.0 <= float(alpha) <= 1.0: raise ValueError("alpha must be in [0,1]")
    if int(ksize)<1 or int(ksize)%2==0: raise ValueError("ksize must be positive odd")
    g0=cv2.getGaussianKernel(int(ksize),float(sigma0),cv2.CV_64F); k0=g0@g0.T; k0/=k0.sum()
    g1=cv2.getGaussianKernel(int(ksize),float(sigma1),cv2.CV_64F); k1=g1@g1.T; k1/=k1.sum()
    kernel=(1.0-float(alpha))*k0+float(alpha)*k1
    x=np.asarray(image,dtype=np.float64)
    if x.ndim==2: return cv2.filter2D(x,cv2.CV_64F,kernel,borderType=cv2.BORDER_REFLECT_101)
    return np.stack([cv2.filter2D(x[:,:,c],cv2.CV_64F,kernel,borderType=cv2.BORDER_REFLECT_101) for c in range(x.shape[2])],axis=2)
