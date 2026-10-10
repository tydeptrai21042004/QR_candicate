from __future__ import annotations

from typing import Any

import numpy as np

from ..core.config import ProposedConfig
from ..core.interfaces import WatermarkMethod
from ..core.types import EmbeddingResult, ExtractionResult
from .embedding import embed_image
from .extraction import extract_image
from .elegant_embedding import embed_elegant_image
from .elegant_extraction import extract_elegant_image
from .blind_embedding import embed_blind_image
from .blind_extraction import extract_blind_image
from .blind_v4_embedding import embed_blind_v4_image
from .blind_v4_extraction import extract_blind_v4_image
from .fused_convqr_v6 import embed_fcqr_v6_image, extract_fcqr_v6_image
from .integer_convolution_v7 import embed_integer_conv8_image, extract_integer_conv8_image
from .integer_convolution_pair_v8 import embed_integer_conv_pair_image, extract_integer_conv_pair_image
from .integer_convolution_quad_v9 import embed_integer_conv_quad_image, extract_integer_conv_quad_image
from .convolution_qr_v5 import embed_convqr_v5_image, extract_convqr_v5_image


class ConvolutionCertifiedR12QIM(WatermarkMethod):
    """QR-QIM family: fully blind v3 plus reproducible v2/v1 research ablations."""

    name = "mecqr_qim_v2"

    def __init__(self, config: ProposedConfig | None = None):
        self.config = config or ProposedConfig()
        self.config.validate()
        if self.config.design == "integer_conv_quad_v9":
            self.name = "blind_integer_convolution_quad_v9_experimental"
        elif self.config.design == "integer_conv_pair_v8":
            self.name = "blind_integer_convolution_pair_v8_experimental"
        elif self.config.design == "integer_conv8_v7":
            self.name = "blind_integer_convolution_v7_experimental"
        elif self.config.design == "fcqr_v6":
            self.name = "blind_fcqr_v6_experimental"
        elif self.config.design == "convqr_v5":
            self.name = "blind_convqr_v5_experimental"
        elif self.config.design == "blind_v4":
            self.name = "blind_cqr_qim_v4"
        elif self.config.design == "blind_v3":
            self.name = "blind_mecqr_qim_v3"
        elif self.config.design == "elegant_v2":
            self.name = "mecqr_qim_v2"
        else:
            self.name = "ccqr_r12_qim_v1"

    def embed(self, host: np.ndarray, watermark: np.ndarray, *, key: bytes) -> EmbeddingResult:
        if self.config.design == "integer_conv_quad_v9":
            image, side, meta = embed_integer_conv_quad_image(host, watermark, key, self.config)
        elif self.config.design == "integer_conv_pair_v8":
            image, side, meta = embed_integer_conv_pair_image(host, watermark, key, self.config)
        elif self.config.design == "integer_conv8_v7":
            image, side, meta = embed_integer_conv8_image(host, watermark, key, self.config)
        elif self.config.design == "fcqr_v6":
            image, side, meta = embed_fcqr_v6_image(host, watermark, key, self.config)
        elif self.config.design == "convqr_v5":
            image, side, meta = embed_convqr_v5_image(host, watermark, key, self.config)
        elif self.config.design == "blind_v4":
            image, side, meta = embed_blind_v4_image(host, watermark, key, self.config)
        elif self.config.design == "blind_v3":
            image, side, meta = embed_blind_image(host, watermark, key, self.config)
        elif self.config.design == "elegant_v2":
            image, side, meta = embed_elegant_image(host, watermark, key, self.config)
        else:
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
        if self.config.design == "integer_conv_quad_v9":
            wm, conf, meta = extract_integer_conv_quad_image(image, key, self.config, watermark_shape)
        elif self.config.design == "integer_conv_pair_v8":
            wm, conf, meta = extract_integer_conv_pair_image(image, key, self.config, watermark_shape)
        elif self.config.design == "integer_conv8_v7":
            wm, conf, meta = extract_integer_conv8_image(image, key, self.config, watermark_shape)
        elif self.config.design == "fcqr_v6":
            wm, conf, meta = extract_fcqr_v6_image(image, key, self.config, watermark_shape)
        elif self.config.design == "convqr_v5":
            wm, conf, meta = extract_convqr_v5_image(image, key, self.config, watermark_shape)
        elif self.config.design == "blind_v4":
            wm, conf, meta = extract_blind_v4_image(image, key, self.config, watermark_shape)
        elif self.config.design == "blind_v3":
            wm, conf, meta = extract_blind_image(image, key, self.config, watermark_shape)
        elif self.config.design == "elegant_v2":
            if side_info is None:
                raise ValueError("ME-CQR-QIM v2 requires authenticated selector side information")
            wm, conf, meta = extract_elegant_image(image, key, side_info, self.config)
        else:
            if side_info is None:
                raise ValueError("CCQR-R12-QIM v1 requires authenticated period side information")
            wm, conf, meta = extract_image(image, key, side_info, self.config)
        return ExtractionResult(watermark=wm, confidence=conf, metadata=meta)


# Backward-compatible import name for experiment scripts written for v2.
ProposedAdaptiveQR = ConvolutionCertifiedR12QIM
