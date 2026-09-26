from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class EmbeddingResult:
    image: np.ndarray
    side_info: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ExtractionResult:
    watermark: np.ndarray
    confidence: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkRecord:
    method: str
    host: str
    watermark: str
    attack: str
    attack_params: str
    seed: int | None
    psnr: float
    ssim: float
    nc: float
    ber: float
    embed_seconds: float
    extract_seconds: float
    confidence: float | None = None


PathLike = str | Path
