from __future__ import annotations

import numpy as np


def canonical_qr(block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    q, r = np.linalg.qr(np.asarray(block, dtype=np.float64))
    signs = np.sign(np.diag(r))
    signs[signs == 0] = 1.0
    d = np.diag(signs)
    q = q @ d
    r = d @ r
    return q, r


def q_angle(q: np.ndarray) -> float:
    """Angle of the first column of a 2x2 orthogonal matrix."""
    return float(np.arctan2(q[1, 0], q[0, 0]))


def q_det_sign(q: np.ndarray) -> float:
    return 1.0 if float(np.linalg.det(q)) >= 0 else -1.0


def q_from_angle(theta: float, det_sign: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    d = 1.0 if det_sign >= 0 else -1.0
    return np.array([[c, -d * s], [s, d * c]], dtype=np.float64)
