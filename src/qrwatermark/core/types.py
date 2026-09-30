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
    # psnr/ssim are retained as backward-compatible aliases for embedding quality.
    psnr: float
    ssim: float
    embedding_psnr: float
    embedding_ssim: float
    attacked_psnr: float
    attacked_ssim: float
    nc: float
    ber: float
    embed_seconds: float
    extract_seconds: float
    confidence: float | None = None
    side_information_bits: int | None = None
    side_information_serialized_bits: int | None = None
    certified_fraction: float | None = None
    certified_ber: float | None = None
    uncertified_ber: float | None = None

PathLike = str | Path
