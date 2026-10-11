"""Offline watermarking stress tests. Never invoked by the production coder.

Off-grid geometric transforms deliberately DO NOT register or invert the warp:
these test synchronization sensitivity rather than claiming invariance.
"""
from __future__ import annotations
import cv2
import numpy as np


def subpixel_translation(image, dx=0.5, dy=1.25):
    h,w=image.shape[:2]
    transform=np.float32([[1,0,float(dx)],[0,1,float(dy)]])
    return cv2.warpAffine(image,transform,(w,h),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def affine_shear(image, shx=0.08, shy=0.0):
    h,w=image.shape[:2]
    m=np.float32([[1.0,shx,-shx*(h-1)*0.5],[shy,1.0,-shy*(w-1)*0.5]])
    return cv2.warpAffine(image,m,(w,h),flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT,borderValue=0)


def aspect_ratio_resample(image, xscale=0.85, yscale=1.1):
    h,w=image.shape[:2]
    distorted=cv2.resize(image,(max(1,round(w*xscale)),max(1,round(h*yscale))),
                         interpolation=cv2.INTER_LINEAR)
    return cv2.resize(distorted,(w,h),interpolation=cv2.INTER_LINEAR)


def green_carrier_null(image, period=66, strength=1.0):
    """Key-independent attack: move EVERY aligned 2x2 green sum to a phase boundary.

    Phase boundaries lie at 0 and T/2 modulo T, halfway between the
    quarter-phase centroids. No watermark bits or secret key are read.  This
    is an adversarial experimental model, not a guaranteed removal attack.
    """
    from ..proposed.green_quad_min_energy import optimal_integer_projection
    if period < 4 or period % 2 or not (0 <= strength <= 1):
        raise ValueError('Even period>=4 and strength in [0,1] required')
    out=image.copy()
    h,w=out.shape[:2];hh=h//2*2;ww=w//2*2
    plane=out[:hh,:ww,1]
    cells=plane.reshape(hh//2,2,ww//2,2).transpose(0,2,1,3).reshape(-1,4)
    sums=cells.astype(np.int32).sum(axis=1)
    # Nearest phase boundary among integer multiples of T/2.
    half=period//2
    target=np.clip(np.rint(sums/half).astype(np.int32)*half,0,1020)
    replacement=optimal_integer_projection(cells,target)
    mixed=np.clip(np.rint((1-strength)*cells+strength*replacement),0,255).astype(np.uint8)
    plane[:]=mixed.reshape(hh//2,ww//2,2,2).transpose(0,2,1,3).reshape(hh,ww)
    return out
