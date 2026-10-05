from __future__ import annotations

"""Validate the single universal blind-v3 method at a target FPS.

This uses configs/methods/proposed.yaml unchanged, including the convolution
certificate.  It measures warm-session end-to-end embedding and blind
extraction over every bundled host/watermark pair and fails when any p95
latency exceeds the requested frame budget.
"""

import argparse
import csv
import time
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrwatermark.core.factory import build_method
from qrwatermark.utils.image_io import read_color
from qrwatermark.utils.watermark import prepare_binary_watermark


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/methods/proposed.yaml")
    ap.add_argument("--repeat", type=int, default=60)
    ap.add_argument("--target-fps", type=float, default=100.0)
    ap.add_argument("--key", default="unified-blind-v3-rt100")
    ap.add_argument("--out", default="validation/unified_full_rt100.csv")
    args = ap.parse_args()

    method = build_method("proposed", ROOT / args.config)
    if method.config.design != "blind_v3":
        raise SystemExit("validation expects blind_v3")
    if not bool(method.config.compute_certificate):
        raise SystemExit(
            "Use the universal proposed.yaml with compute_certificate=true; "
            "this validator intentionally includes convolution certification."
        )

    target_ms = 1000.0 / float(args.target_fps)
    key = args.key.encode("utf-8")
    hosts = sorted((ROOT / "data" / "hosts" / "classical").glob("*.bmp"))
    watermarks = sorted((ROOT / "data" / "watermarks").glob("*.png"))
    rows: list[dict[str, object]] = []

    for hp in hosts:
        host = read_color(hp)
        for wp in watermarks:
            wm = prepare_binary_watermark(wp, method.config.watermark_size)

            # Warm key/geometry/Arnold caches and OpenCV kernels.
            emb = method.embed(host, wm, key=key)
            method.extract(emb.image, key=key, side_info=None, watermark_shape=wm.shape)

            embed_ms = []
            extract_ms = []
            for _ in range(max(1, int(args.repeat))):
                t0 = time.perf_counter()
                emb = method.embed(host, wm, key=key)
                embed_ms.append((time.perf_counter() - t0) * 1000.0)

                t0 = time.perf_counter()
                ext = method.extract(
                    emb.image, key=key, side_info=None, watermark_shape=wm.shape
                )
                extract_ms.append((time.perf_counter() - t0) * 1000.0)

            emed = float(np.median(embed_ms))
            ep95 = float(np.quantile(embed_ms, 0.95))
            xmed = float(np.median(extract_ms))
            xp95 = float(np.quantile(extract_ms, 0.95))
            clean_exact = bool(np.array_equal(ext.watermark, wm))
            row = {
                "host": hp.stem,
                "watermark": wp.stem,
                "embed_median_ms": emed,
                "embed_p95_ms": ep95,
                "embed_median_fps": 1000.0 / emed,
                "embed_p95_fps": 1000.0 / ep95,
                "extract_median_ms": xmed,
                "extract_p95_ms": xp95,
                "extract_median_fps": 1000.0 / xmed,
                "extract_p95_fps": 1000.0 / xp95,
                "psnr_db": float(emb.metadata["actual_psnr_db"]),
                "certified_fraction": float(emb.metadata["certified_fraction"]),
                "certificate_computed": bool(emb.metadata["certificate_computed"]),
                "clean_exact": clean_exact,
                "embed_p95_pass": ep95 <= target_ms,
                "extract_p95_pass": xp95 <= target_ms,
            }
            rows.append(row)
            print(
                f"{hp.stem}/{wp.stem}: embed p95={ep95:.3f} ms "
                f"({1000.0/ep95:.1f} FPS), extract p95={xp95:.3f} ms, "
                f"PSNR={row['psnr_db']:.3f} dB, cert={row['certified_fraction']:.3f}"
            )

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    worst_embed = max(float(r["embed_p95_ms"]) for r in rows)
    worst_extract = max(float(r["extract_p95_ms"]) for r in rows)
    all_ok = all(
        bool(r["embed_p95_pass"])
        and bool(r["extract_p95_pass"])
        and bool(r["clean_exact"])
        and bool(r["certificate_computed"])
        for r in rows
    )
    print("--- aggregate ---")
    print(f"pairs={len(rows)}")
    print(f"target={args.target_fps:g} FPS -> {target_ms:.3f} ms/frame")
    print(f"worst embed p95={worst_embed:.3f} ms ({1000.0/worst_embed:.1f} FPS)")
    print(f"worst extract p95={worst_extract:.3f} ms ({1000.0/worst_extract:.1f} FPS)")
    print(f"full convolution certificate enabled={bool(method.config.compute_certificate)}")
    print(f"PASS={all_ok}")
    if not all_ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
