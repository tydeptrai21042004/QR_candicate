from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path

import pandas as pd

from _common import load_yaml, resolve
from qrwatermark.core.config import load_yaml as load_core_yaml, proposed_from_dict
from qrwatermark.evaluation.evaluator import embed_once, evaluate_embedded
from qrwatermark.proposed import ProposedAdaptiveQR
from qrwatermark.utils.image_io import read_color
from qrwatermark.utils.watermark import prepare_binary_watermark


def deep_update(base, patch):
    out = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def _attack_instances(attacks):
    for spec in attacks:
        name = str(spec["name"])
        params = dict(spec.get("params", {}))
        seeds = spec.get("seeds")
        if seeds is None:
            yield name, params, None
        else:
            for seed in seeds:
                yield name, params, int(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/ablation.yaml")
    ap.add_argument("--key", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--variant", action="append", default=None,
                    help="Run only a named variant; repeat this flag for multiple variants")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    base = load_core_yaml(resolve(cfg["base_config"]))["parameters"]
    watermark_size = int(cfg.get("watermark_size", base.get("watermark_size", 64)))

    variants = cfg["variants"]
    if args.variant:
        unknown = [v for v in args.variant if v not in variants]
        if unknown:
            ap.error(f"unknown variant(s): {', '.join(unknown)}")
        variants = {v: variants[v] for v in args.variant}

    host_paths = sorted(resolve(cfg.get("hosts", "data/hosts/classical")).glob("*.bmp"))
    watermark_paths = sorted(resolve(cfg.get("watermarks", "data/watermarks")).glob("*.png"))
    attack_cfg = load_yaml(resolve(cfg.get("attacks", "configs/attacks/standard.yaml")))
    attacks = list(_attack_instances(attack_cfg["attacks"]))

    rows = []
    key = args.key.encode("utf-8")
    for variant_name, patch in variants.items():
        method_cfg = proposed_from_dict(deep_update(base, patch or {}))
        method = ProposedAdaptiveQR(method_cfg)
        for wp in watermark_paths:
            wm = prepare_binary_watermark(wp, watermark_size)
            for hp in host_paths:
                host = read_color(hp)
                # Embed once per host/watermark/variant and reuse the identical
                # watermarked image for every attack, matching the main benchmark.
                emb, embed_s = embed_once(method, host, wm, key=key)
                for attack_name, attack_params, seed in attacks:
                    rec, _ = evaluate_embedded(
                        method, host, wm, emb, embed_seconds=embed_s, key=key,
                        host_name=hp.stem, watermark_name=wp.stem,
                        attack_name=attack_name, attack_params=attack_params, seed=seed,
                    )
                    row = rec.__dict__.copy()
                    row["variant"] = variant_name
                    rows.append(row)

    rd = Path(args.run_dir)
    rd.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(rd / "ablation.csv", index=False)

    metric_cols = [
        c for c in ("embedding_psnr", "embedding_ssim", "nc", "ber", "certified_fraction",
                    "embed_seconds", "extract_seconds", "side_information_bits")
        if c in df.columns
    ]
    summary = (
        df.groupby(["variant", "attack"], dropna=False)[metric_cols]
          .agg(["mean", "std", "count"])
          .reset_index()
    )
    summary.columns = [
        "_".join(str(x) for x in col if x != "").rstrip("_") if isinstance(col, tuple) else col
        for col in summary.columns
    ]
    summary.to_csv(rd / "ablation_summary.csv", index=False)

    overall = df.groupby("variant", dropna=False)[metric_cols].mean(numeric_only=True).reset_index()
    overall.to_csv(rd / "ablation_overall.csv", index=False)
    print(f"Completed {len(df)} evaluations across {len(variants)} variants -> {rd}")
    print(f"Raw: {rd / 'ablation.csv'}")
    print(f"Per-attack summary: {rd / 'ablation_summary.csv'}")
    print(f"Overall means: {rd / 'ablation_overall.csv'}")


if __name__ == "__main__":
    main()
