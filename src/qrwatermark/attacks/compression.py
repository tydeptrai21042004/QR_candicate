from __future__ import annotations

import cv2
import numpy as np


def jpeg(image: np.ndarray, quality: int = 50) -> np.ndarray:
    ok, enc = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
    if not ok:
        raise RuntimeError('JPEG encoding failed')
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if out is None:
        raise RuntimeError('JPEG decoding failed')
    return out


def jpeg2000(image: np.ndarray, compression_ratio: float = 13.0) -> np.ndarray:
    # OpenCV exposes JPEG2000 compression via IMWRITE_JPEG2000_COMPRESSION_X1000.
    # Convert ratio to a conservative quality-like target; callers should report this mapping.
    ratio = max(float(compression_ratio), 1.0)
    x1000 = int(max(1, min(1000, round(1000.0 / ratio))))
    ok, enc = cv2.imencode('.jp2', image, [cv2.IMWRITE_JPEG2000_COMPRESSION_X1000, x1000])
    if not ok:
        raise RuntimeError('JPEG2000 encoding failed; OpenCV build may lack JP2 support')
    out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    if out is None:
        raise RuntimeError('JPEG2000 decoding failed')
    return out
