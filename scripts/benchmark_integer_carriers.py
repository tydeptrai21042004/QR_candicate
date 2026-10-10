"""Run reproducible integer-carrier comparisons without the absent FCQR-v6 script.

Usage (from repository root, after ``pip install -e .``)::

    python scripts/benchmark_integer_carriers.py --out evidence_integer_carriers
    python scripts/benchmark_integer_carriers.py --out /tmp/qr-smoke --smoke

Compares the available proposal-family methods v5, v7, v8 and v9.
FCQR-v6 is optional only if its authentic source is supplied.
For comparisons against published baselines use scripts/run_main_comparison.py.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import tracemalloc
from pathlib import Path

import cv2
import pandas as pd

from _common import load_yaml
from qrwatermark.core.factory import build_method
from qrwatermark.evaluation.benchmark import run_benchmark
from qrwatermark.utils.image_io import read_color
from qrwatermark.utils.watermark import prepare_binary_watermark

ROOT = Path(__file__).resolve().parents[1]


def attacks_26():
    items = [
        {"name": "clean"},
        {"name": "gaussian_blur", "params": {"sigma": 0.7}},
        {"name": "gaussian_blur", "params": {"sigma": 1.0}},
        {"name": "lowpass", "params": {"kx": 5, "ky": 5}},
        {"name": "average", "params": {"ksize": 3}},
        {"name": "motion_blur", "params": {"ksize": 5}},
        {"name": "jpeg", "params": {"quality": 90}},
        {"name": "jpeg", "params": {"quality": 70}},
        {"name": "jpeg", "params": {"quality": 50}},
    ]
    for v in (0.001, 0.003):
        for seed in (0, 1, 2):
            items.append({"name": "gaussian_noise", "params": {"variance": v}, "seeds": [seed]})
    for density in (0.02, 0.05):
        for seed in (0, 1, 2):
            items.append({"name": "salt_pepper", "params": {"density": density}, "seeds": [seed]})
    items.extend([
        {"name": "scale_resample", "params": {"scale": 0.8}},
        {"name": "registered_rotation_resample", "params": {"angle": 5}},
        {"name": "rotation_unregistered", "params": {"angle": 2}},
        {"name": "translation", "params": {"dx": 2, "dy": 2}},
        {"name": "crop_resize", "params": {"fraction": 0.05}},
    ])
    assert len(items) == 26
    return items


def timing(method, host, wm, key, repeats):
    image = method.embed(host, wm, key=key).image
    results = {}
    for name, fn in (
        ("embed", lambda: method.embed(host, wm, key=key)),
        ("extract", lambda: method.extract(image, key=key)),
    ):
        fn()  # warm-up
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            fn()
            times.append(time.perf_counter() - start)
        tracemalloc.start()
        fn()
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        results[name + "_median_ms"] = statistics.median(times) * 1000
        # Python allocations only, not a full native/OpenCV resident-set-size measurement.
        results[name + "_python_peak_mib"] = peak / (1024 * 1024)
    results["fps_embed_plus_extract"] = 1000 / (
        results["embed_median_ms"] + results["extract_median_ms"]
    )
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("evidence_integer_carriers"))
    ap.add_argument("--repeats", type=int, default=15)
    ap.add_argument("--smoke", action="store_true", help="One image/watermark and clean attack only")
    ap.add_argument("--include-v6", action="store_true", help="Require authentic FCQR-v6 implementation")
    ap.add_argument("--attack-config", type=Path, default=None, help="Optional attack YAML override")
    ap.add_argument("--key", default="fixed-reproducibility-key-20261010")
    args = ap.parse_args()
    if args.repeats < 1:
        ap.error("--repeats must be >= 1")
    cv2.setNumThreads(1)
    methods = [
        ("ConvQR-v5", "proposed_convqr_v5_experimental.yaml"),
        ("IntegerConv8-v7", "proposed_integer_conv8_v7_experimental.yaml"),
        ("IntegerConvPair-v8", "proposed_integer_conv_pair_v8_experimental.yaml"),
        ("IntegerConvQuad-v9", "proposed_integer_conv_quad_v9_experimental.yaml"),
    ]
    if args.include_v6:
        if not (ROOT / "src/qrwatermark/proposed/fused_convqr_v6.py").is_file():
            ap.error("FCQR-v6 is unavailable. Restore its authentic source before --include-v6")
        methods.insert(1, ("FCQR-v6", "proposed_fcqr_v6_experimental.yaml"))
    paths = [(label, ROOT / "configs/methods" / cfg) for label, cfg in methods]
    missing = [str(p) for _, p in paths if not p.is_file()]
    if missing:
        ap.error("missing method configuration(s): " + ", ".join(missing))
    methods = [(label, build_method("proposed", str(p))) for label, p in paths]
    hosts = sorted((ROOT / "data/hosts/classical").glob("*.bmp"))
    wms = sorted((ROOT / "data/watermarks").glob("*.png"))
    if not hosts or not wms:
        ap.error("host/watermark images are missing")
    if args.smoke:
        hosts, wms = hosts[:1], wms[:1]
    attacks = ([{"name": "clean"}] if args.smoke else (
        load_yaml(str(args.attack_config))["attacks"] if args.attack_config else attacks_26()
    ))
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    df = run_benchmark([m for _, m in methods], hosts, wms, attacks,
                       key=args.key.encode(), run_dir=out,
                       watermark_size=64, match_psnr_to_proposed=False)
    summary = {}
    for label, method in methods:
        d = df[df["method"] == method.name]
        if d.empty:
            raise RuntimeError(f"no rows for {label}")
        clean = d[d.attack == "clean"]
        attacked = d[d.attack != "clean"]
        summary[label] = {
            "method_id": method.name,
            "host_count": len(hosts), "watermark_count": len(wms),
            "attacks_per_pair": len(attacks) - 1,
            "mean_clean_psnr_db": float(clean.embedding_psnr.mean()),
            "max_clean_ber": float(clean.ber.max()),
            "mean_attacked_nc": float(attacked.nc.mean()) if not attacked.empty else None,
            "mean_attacked_ber": float(attacked.ber.mean()) if not attacked.empty else None,
        }
    host = read_color(hosts[0]); wm = prepare_binary_watermark(wms[0], 64)
    runtimes = []
    for label, method in methods:
        row = {"method": label, **timing(method, host, wm, args.key.encode(), args.repeats)}
        runtimes.append(row)
        summary[label]["runtime"] = row
    pd.DataFrame(runtimes).to_csv(out / "runtime_carriers.csv", index=False)
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved {len(df)} evaluations and runtimes to {out}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
