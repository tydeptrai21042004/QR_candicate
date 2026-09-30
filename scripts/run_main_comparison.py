from __future__ import annotations

import argparse
from pathlib import Path

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
    methods = [build_method(m["name"], resolve(m["config"])) for m in cfg["methods"]]
    hosts = sorted(resolve(cfg["hosts"]).glob("*.bmp"))
    watermarks = sorted(resolve(cfg["watermarks"]).glob("*.png"))
    attacks = load_yaml(cfg["attacks"])["attacks"]
    df = run_benchmark(methods, hosts, watermarks, attacks, key=args.key.encode(), run_dir=Path(args.run_dir), save_images=args.save_images, match_psnr_to_proposed=bool(cfg.get("match_psnr_to_proposed", True)))
    aggregate_records(df).to_csv(Path(args.run_dir) / "summary.csv", index=False)
    print(f"Completed {len(df)} evaluations -> {args.run_dir}")


if __name__ == "__main__":
    main()
