"""Single proposed watermark method. Literature baselines remain separate."""
from __future__ import annotations
from typing import Any
import numpy as np
from ..core.config import ProposedConfig
from ..core.interfaces import WatermarkMethod
from ..core.types import EmbeddingResult,ExtractionResult
from .green_quad_min_energy import embed_green_quad_min_energy_image,extract_green_quad_min_energy_image

class GreenQuadWatermark(WatermarkMethod):
    name='green_quad_min_energy'
    def __init__(self, config: ProposedConfig | None=None):
        self.config=config or ProposedConfig()
        self.config.validate()
    def embed(self,host:np.ndarray,watermark:np.ndarray,*,key:bytes)->EmbeddingResult:
        image,side,meta=embed_green_quad_min_energy_image(host,watermark,key,self.config)
        return EmbeddingResult(image=image,side_info=side,metadata=meta)
    def extract(self,image:np.ndarray,*,key:bytes,side_info:dict[str,Any]|None=None,watermark_shape:tuple[int,int]=(64,64))->ExtractionResult:
        if side_info is not None:raise ValueError('fully blind method never accepts side information')
        wm,conf,meta=extract_green_quad_min_energy_image(image,key,self.config,watermark_shape)
        return ExtractionResult(watermark=wm,confidence=conf,metadata=meta)
