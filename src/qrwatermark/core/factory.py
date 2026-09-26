from __future__ import annotations

from pathlib import Path

from .config import baseline_from_dict, load_yaml, proposed_from_dict
from ..proposed import ConvolutionCertifiedR12QIM
from ..baselines import Nha2022ImprovedQR, Su2017Hessenberg, Su2020Schur


def build_method(name: str, config_path: str | Path | None = None):
    data = load_yaml(config_path) if config_path else {}
    if "method" in data:
        data = dict(data.get("parameters", {}))
    name = name.lower()
    if name in {"proposed", "ccqr", "ccqr_r12_qim_v1", "proposed_adaptive_qr_v2"}:
        return ConvolutionCertifiedR12QIM(proposed_from_dict(data))
    cfg = baseline_from_dict(data)
    if name in {"su2017", "su2017_hessenberg"}:
        return Su2017Hessenberg(cfg)
    if name in {"su2020", "su2020_schur"}:
        return Su2020Schur(cfg)
    if name in {"nha2022", "nha2022_improved_qr"}:
        return Nha2022ImprovedQR(cfg)
    raise ValueError(f"Unknown method: {name}")
