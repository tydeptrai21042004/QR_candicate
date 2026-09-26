from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class NLMConfig:
    enabled: bool = True
    compare_always: bool = True
    mild_h: float = 4.0
    mild_template: int = 3
    mild_search: int = 11
    strong_h: float = 8.0
    strong_template: int = 5
    strong_search: int = 17


@dataclass
class ProposedConfig:
    block_size: int = 2
    channel: int = 0  # OpenCV BGR: 0 = blue
    watermark_size: int = 64
    arnold_iterations: int = 10
    repetition: int = 5
    q_angle_period: float = 0.12
    r12_period: float = 8.0
    distortion_scale: float = 1.0
    lambda_distortion: float = 0.35
    utility_imbalance_penalty: float = 0.15
    selector_group_mean: bool = True
    perturb_plus_minus: float = 1.0
    perturb_blur_sigma: float = 0.6
    nlm: NLMConfig = field(default_factory=NLMConfig)

    def validate(self) -> None:
        if self.block_size != 2:
            raise ValueError("The revised dual-branch proposal is derived for 2x2 blocks.")
        if self.repetition < 1 or self.repetition % 2 == 0:
            raise ValueError("repetition must be a positive odd integer")
        if self.q_angle_period <= 0 or self.r12_period <= 0:
            raise ValueError("Q/R QIM periods must be positive")
        if self.distortion_scale <= 0:
            raise ValueError("distortion_scale must be positive")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BaselineConfig:
    block_size: int = 4
    channel: int = 0
    watermark_size: int = 64
    arnold_iterations: int = 10
    quant_step: float = 8.0
    threshold: float = 0.04


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        obj = yaml.safe_load(f)
    return obj or {}


def proposed_from_dict(data: dict[str, Any]) -> ProposedConfig:
    d = dict(data)
    nlm = d.pop("nlm", None)
    cfg = ProposedConfig(**d)
    if nlm is not None:
        cfg.nlm = NLMConfig(**nlm)
    cfg.validate()
    return cfg


def baseline_from_dict(data: dict[str, Any]) -> BaselineConfig:
    return BaselineConfig(**data)
