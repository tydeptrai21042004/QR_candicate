from __future__ import annotations
import numpy as np


def random_pixel_dropout(image:np.ndarray,fraction:float=0.5,seed:int|None=3,value:int=0)->np.ndarray:
    """Independent random pixel dropout (not a contiguous occlusion)."""
    rng=np.random.default_rng(seed); out=image.copy(); mask=rng.random(image.shape[:2])<float(fraction); out[mask]=int(value); return out


def rectangular_occlusion(image:np.ndarray,fraction:float=0.25,seed:int|None=3,value:int=0)->np.ndarray:
    """Cover one contiguous rectangle whose area is approximately `fraction`."""
    if not 0.0<=float(fraction)<1.0: raise ValueError("fraction must be in [0,1)")
    rng=np.random.default_rng(seed); out=image.copy(); h,w=image.shape[:2]
    side=float(np.sqrt(fraction)); rh=max(1,min(h,int(round(h*side)))); rw=max(1,min(w,int(round(w*side))))
    r=0 if rh==h else int(rng.integers(0,h-rh+1)); c=0 if rw==w else int(rng.integers(0,w-rw+1))
    out[r:r+rh,c:c+rw]=int(value); return out


# Backward-compatible alias.  New experiment configs use random_pixel_dropout.
random_occlusion=random_pixel_dropout
