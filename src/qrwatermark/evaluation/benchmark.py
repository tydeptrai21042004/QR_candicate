from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from ..core.interfaces import WatermarkMethod
from ..utils.image_io import read_color, write_image
from ..utils.watermark import prepare_binary_watermark
from .evaluator import evaluate_once


def run_benchmark(
    methods: Iterable[WatermarkMethod],
    host_paths: list[Path],
    watermark_paths: list[Path],
    attacks: list[dict],
    *,
    key: bytes,
    run_dir: Path,
    save_images: bool = False,
) -> pd.DataFrame:
    run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for wm_path in watermark_paths:
        wm = prepare_binary_watermark(wm_path, 64)
        for host_path in host_paths:
            host = read_color(host_path)
            for method in methods:
                for attack in attacks:
                    name = attack.get("name", "clean")
                    params = dict(attack.get("params", {}))
                    seeds = attack.get("seeds", [None])
                    for seed in seeds:
                        rec, extra = evaluate_once(
                            method, host, wm, key=key,
                            host_name=host_path.stem,
                            watermark_name=wm_path.stem,
                            attack_name=name,
                            attack_params=params,
                            seed=seed,
                        )
                        records.append(rec.__dict__)
                        if save_images:
                            stem = f"{method.name}__{host_path.stem}__{wm_path.stem}__{name}__seed{seed}"
                            write_image(run_dir / "images" / f"{stem}__watermarked.png", extra["watermarked"])
                            write_image(run_dir / "images" / f"{stem}__attacked.png", extra["attacked"])
                            write_image(run_dir / "images" / f"{stem}__extracted.png", extra["extracted"])
    df = pd.DataFrame.from_records(records)
    df.to_csv(run_dir / "metrics.csv", index=False)
    return df
