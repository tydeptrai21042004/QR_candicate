from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class ConvolutionKernel:
    name: str
    kernel: np.ndarray


def gaussian_kernel(size: int, sigma: float) -> np.ndarray:
    if size < 1 or size % 2 == 0:
        raise ValueError("kernel size must be a positive odd integer")
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    g = cv2.getGaussianKernel(int(size), float(sigma), cv2.CV_64F)
    k = g @ g.T
    return k / float(k.sum())


def convolution_bank(kernel_size: int, gaussian_sigmas: tuple[float, ...] | list[float]) -> list[ConvolutionKernel]:
    return [ConvolutionKernel(f"gaussian_sigma_{float(s):g}", gaussian_kernel(kernel_size, float(s))) for s in gaussian_sigmas]


def kernel_transfer_function(kernel: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """DFT multiplier for circular convolution on Z_M x Z_N.

    The kernel's geometric centre is shifted to the origin before the DFT, so
    multiplying fft2(image) by the returned array implements the finite-torus
    convolution model used by the theoretical analysis.
    """
    k = np.asarray(kernel, dtype=np.float64)
    h, w = (int(shape[0]), int(shape[1]))
    if k.ndim != 2 or k.shape[0] > h or k.shape[1] > w:
        raise ValueError("kernel must be 2-D and no larger than the image")
    padded = np.zeros((h, w), dtype=np.float64)
    kh, kw = k.shape
    padded[:kh, :kw] = k
    padded = np.roll(padded, -(kh // 2), axis=0)
    padded = np.roll(padded, -(kw // 2), axis=1)
    return np.fft.fft2(padded)


def circular_convolve(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float64)
    transfer = kernel_transfer_function(kernel, x.shape)
    return np.fft.ifft2(np.fft.fft2(x) * transfer).real


def spectral_distance_from_identity(kernel: np.ndarray, shape: tuple[int, int]) -> float:
    """Return ||T_h-I||_{2->2} = max_omega |h_hat(omega)-1|."""
    transfer = kernel_transfer_function(kernel, shape)
    return float(np.max(np.abs(transfer - 1.0)))
