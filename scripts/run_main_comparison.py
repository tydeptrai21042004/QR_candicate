from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import replace

from _common import load_yaml, resolve
from qrwatermark.core.factory import build_method
from qrwatermark.evaluation.aggregation import aggregate_records
from qrwatermark.evaluation.benchmark import run_benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/main_comparison.yaml")
    ap.add_argument("--key", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--save-images", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    watermark_size = int(cfg.get("watermark_size", 64))
    methods = [build_method(m["name"], resolve(m["config"])) for m in cfg["methods"]]
    # Experiment-level payload size overrides the method YAML so a small-payload
    # control study (e.g. Zareian2013: one bit per 16x16 host block) remains an
    # apples-to-apples comparison across every method.
    for method in methods:
        if hasattr(method, "config") and hasattr(method.config, "watermark_size"):
            method.config = replace(method.config, watermark_size=watermark_size)
    hosts = sorted(resolve(cfg["hosts"]).glob("*.bmp"))
    watermarks = sorted(resolve(cfg["watermarks"]).glob("*.png"))
    attacks = load_yaml(cfg["attacks"])["attacks"]
    df = run_benchmark(
        methods, hosts, watermarks, attacks,
        key=args.key.encode(),
        run_dir=Path(args.run_dir),
        save_images=args.save_images,
        match_psnr_to_proposed=bool(cfg.get("match_psnr_to_proposed", True)),
        watermark_size=watermark_size,
    )
    aggregate_records(df).to_csv(Path(args.run_dir) / "summary.csv", index=False)
    print(f"Completed {len(df)} evaluations -> {args.run_dir}")


if __name__ == "__main__":
    main()
