from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd

from ..core.interfaces import WatermarkMethod
from ..proposed.method import ConvolutionCertifiedR12QIM
from ..utils.image_io import read_color, write_image
from ..utils.watermark import prepare_binary_watermark
from .evaluator import embed_once, evaluate_embedded
from .fairness import tune_baseline_to_psnr
from .metrics import psnr


def _evaluate_attacks(
    method,
    host,
    wm,
    emb,
    embed_s,
    attacks,
    key,
    host_name,
    wm_name,
    run_dir,
    save_images,
    records,
    tuning=None,
):
    if save_images:
        stem = f"{method.name}__{host_name}__{wm_name}"
        write_image(run_dir / "images" / f"{stem}__watermarked.png", emb.image)
    for attack in attacks:
        name = attack.get("name", "clean")
        params = dict(attack.get("params", {}))
        seeds = attack.get("seeds", [None])
        for seed in seeds:
            rec, extra = evaluate_embedded(
                method,
                host,
                wm,
                emb,
                embed_seconds=embed_s,
                key=key,
                host_name=host_name,
                watermark_name=wm_name,
                attack_name=name,
                attack_params=params,
                seed=seed,
            )
            row = rec.__dict__.copy()
            if tuning:
                row["matched_psnr_target_db"] = tuning.get("target_psnr_db")
                row["achieved_embedding_psnr_db"] = tuning.get("achieved_psnr_db")
                row["strength_scale"] = tuning.get("strength_scale")
                row["strength_fields"] = tuning.get("strength_fields")
                row["tuned_quant_step"] = tuning.get("quant_step")
                row["tuned_threshold"] = tuning.get("threshold")
                row["tuning_seconds"] = tuning.get("tuning_seconds", 0.0)
                row["matched_psnr_error_db"] = tuning.get("psnr_error_db", 0.0)
            records.append(row)
            if save_images:
                astem = f"{method.name}__{host_name}__{wm_name}__{name}__seed{seed}"
                write_image(run_dir / "images" / f"{astem}__attacked.png", extra["attacked"])
                write_image(run_dir / "images" / f"{astem}__extracted.png", extra["extracted"])


def run_benchmark(
    methods: Iterable[WatermarkMethod],
    host_paths: list[Path],
    watermark_paths: list[Path],
    attacks: list[dict],
    *,
    key: bytes,
    run_dir: Path,
    save_images: bool = False,
    match_psnr_to_proposed: bool = False,
    watermark_size: int = 64,
) -> pd.DataFrame:
    """Cache embeddings; optionally tune each paper baseline to proposal PSNR."""
    run_dir.mkdir(parents=True, exist_ok=True)
    records = []
    methods = list(methods)
    for wm_path in watermark_paths:
        wm = prepare_binary_watermark(wm_path, int(watermark_size))
        for host_path in host_paths:
            host = read_color(host_path)
            if match_psnr_to_proposed:
                proposed = next((m for m in methods if isinstance(m, ConvolutionCertifiedR12QIM)), None)
                if proposed is None:
                    raise ValueError("match_psnr_to_proposed requires the proposed method")
                p_emb, p_time = embed_once(proposed, host, wm, key=key)
                target = float(psnr(host, p_emb.image))
                _evaluate_attacks(
                    proposed,
                    host,
                    wm,
                    p_emb,
                    p_time,
                    attacks,
                    key,
                    host_path.stem,
                    wm_path.stem,
                    run_dir,
                    save_images,
                    records,
                    {
                        "target_psnr_db": target,
                        "achieved_psnr_db": target,
                        "strength_scale": 1.0,
                        "psnr_error_db": 0.0,
                    },
                )
                for method in methods:
                    if method is proposed:
                        continue
                    tuned, emb, diag = tune_baseline_to_psnr(
                        method, host, wm, key=key, target_psnr_db=target
                    )
                    _evaluate_attacks(
                        tuned,
                        host,
                        wm,
                        emb,
                        float(diag.get("embed_seconds", 0.0)),
                        attacks,
                        key,
                        host_path.stem,
                        wm_path.stem,
                        run_dir,
                        save_images,
                        records,
                        diag,
                    )
            else:
                for method in methods:
                    emb, embed_s = embed_once(method, host, wm, key=key)
                    _evaluate_attacks(
                        method,
                        host,
                        wm,
                        emb,
                        embed_s,
                        attacks,
                        key,
                        host_path.stem,
                        wm_path.stem,
                        run_dir,
                        save_images,
                        records,
                    )
    df = pd.DataFrame.from_records(records)
    df.to_csv(run_dir / "metrics.csv", index=False)
    return df
