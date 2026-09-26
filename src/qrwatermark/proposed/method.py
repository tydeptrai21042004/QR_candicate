from __future__ import annotations

from typing import Any

import numpy as np

from ..core.config import ProposedConfig
from ..core.interfaces import WatermarkMethod
from ..core.types import EmbeddingResult, ExtractionResult
from .embedding import embed_image
from .extraction import extract_image


class ProposedAdaptiveQR(WatermarkMethod):
    name = "proposed_adaptive_qr_v2"

    def __init__(self, config: ProposedConfig | None = None):
        self.config = config or ProposedConfig()
        self.config.validate()

    def embed(self, host: np.ndarray, watermark: np.ndarray, *, key: bytes) -> EmbeddingResult:
        image, side, meta = embed_image(host, watermark, key, self.config)
        return EmbeddingResult(image=image, side_info=side, metadata=meta)

    def extract(
        self,
        image: np.ndarray,
        *,
        key: bytes,
        side_info: dict[str, Any] | None = None,
        watermark_shape: tuple[int, int] = (64, 64),
    ) -> ExtractionResult:
        if side_info is None:
            raise ValueError("The proposed semi-blind method requires authenticated side information")
        wm, conf, meta = extract_image(image, key, side_info, self.config)
        return ExtractionResult(watermark=wm, confidence=conf, metadata=meta)
