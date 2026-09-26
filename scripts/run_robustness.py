from __future__ import annotations

import argparse
from pathlib import Path

from _common import load_yaml, resolve
from qrwatermark.core.factory import build_method
from qrwatermark.evaluation.aggregation import aggregate_records
from qrwatermark.evaluation.benchmark import run_benchmark


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/experiments/robustness.yaml")
    ap.add_argument("--key", required=True)
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    m = cfg["method"]
    method = build_method(m["name"], resolve(m["config"]))
    hosts = sorted(resolve(cfg["hosts"]).glob("*.bmp"))
    watermarks = sorted(resolve(cfg["watermarks"]).glob("*.png"))
    attacks = load_yaml(cfg["attacks"])["attacks"]
    df = run_benchmark([method], hosts, watermarks, attacks, key=args.key.encode(), run_dir=Path(args.run_dir))
    aggregate_records(df).to_csv(Path(args.run_dir) / "summary.csv", index=False)
    print(f"Completed {len(df)} evaluations")

if __name__ == "__main__": main()
