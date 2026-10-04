from __future__ import annotations

from pathlib import Path

from .config import baseline_from_dict, load_yaml, proposed_from_dict
from ..proposed import ConvolutionCertifiedR12QIM
from ..baselines import (
    Chen2021QuaternionQR,
    Nha2022ImprovedQR,
    Su2014QR,
    Su2016Hessenberg,
    Su2020Schur,
    Zareian2013AdaptiveQIM,
)


def build_method(name: str, config_path: str | Path | None = None):
    data = load_yaml(config_path) if config_path else {}
    if "method" in data:
        data = dict(data.get("parameters", {}))
    name = name.lower()
    if name in {"proposed", "ccqr", "mecqr_qim_v2", "ccqr_r12_qim_v1", "proposed_adaptive_qr_v2", "blind_mecqr_qim_v3", "blind_v3"}:
        return ConvolutionCertifiedR12QIM(proposed_from_dict(data))
    cfg = baseline_from_dict(data)
    if name in {"su2014", "su2014_qr"}:
        return Su2014QR(cfg)
    if name in {"su2016", "su2016_hessenberg"}:
        return Su2016Hessenberg(cfg)
    if name in {"chen2021", "chen2021_qqrd", "chen2021_quaternion_qr"}:
        return Chen2021QuaternionQR(cfg)
    if name in {"su2020", "su2020_schur"}:
        return Su2020Schur(cfg)
    if name in {"nha2022", "nha2022_improved_qr"}:
        return Nha2022ImprovedQR(cfg)
    if name in {"zareian2013", "zareian2013_aqim", "adaptive_qim"}:
        return Zareian2013AdaptiveQIM(cfg)
    raise ValueError(f"Unknown method: {name}")
