from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from .types import EmbeddingResult, ExtractionResult


class WatermarkMethod(ABC):
    """Common interface used by the proposed method and every baseline."""

    name: str = "method"

    @abstractmethod
    def embed(self, host: np.ndarray, watermark: np.ndarray, *, key: bytes) -> EmbeddingResult:
        raise NotImplementedError

    @abstractmethod
    def extract(
        self,
        image: np.ndarray,
        *,
        key: bytes,
        side_info: dict[str, Any] | None = None,
        watermark_shape: tuple[int, int] = (64, 64),
    ) -> ExtractionResult:
        raise NotImplementedError
