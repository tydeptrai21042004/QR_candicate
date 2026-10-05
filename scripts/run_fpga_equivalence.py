from __future__ import annotations

"""Validate the FPGA-oriented path without changing the watermark algorithm.

Checks:
1. the new streaming software path is bit-for-bit identical to the RT100
   matrix reference for every bundled host/watermark pair;
2. the integer Q18 embed reference reproduces the same two modified pixels;
3. the Q18 decoder reproduces the floating decoder under every configured
   standard attack instance.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrwatermark.attacks.registry import apply_attack
from qrwatermark.core.factory import build_method
from qrwatermark.proposed.streaming_datapath import (
    DEFAULT_FPGA_FRAC_BITS,
    project_safe_block_q18,
    selected_r12_q18_bits,
)
from qrwatermark.utils.image_io import read_color
from qrwatermark.utils.permutation import selected_block_arrays
from qrwatermark.utils.watermark import (
    prepare_binary_watermark,
    scrambled_bits_from_watermark,
    watermark_from_scrambled_bits,
)


def _attack_instances(path: Path):
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    out = []
    for item in cfg.get("attacks", []):
        name = item["name"]
        params = dict(item.get("params", {}))
        seeds = item.get("seeds")
        if seeds is None:
            out.append((name, params))
        else:
            for seed in seeds:
                p = dict(params)
                p["seed"] = int(seed)
                out.append((name, p))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/methods/proposed.yaml")
    ap.add_argument("--attacks", default="configs/attacks/standard.yaml")
    ap.add_argument("--key", default="blind-v3-fpga-equivalence")
    ap.add_argument("--out", default="validation/fpga_q18_equivalence.csv")
    args = ap.parse_args()

    method = build_method("proposed", ROOT / args.config)
    if float(method.config.qim_period) != 48.0 or float(method.config.blind_margin_ratio) != 0.125:
        raise SystemExit("Q18 deployment reference is specialized to Delta=48, margin=1/8")

    key = args.key.encode("utf-8")
    hosts = sorted((ROOT / "data" / "hosts" / "classical").glob("*.bmp"))
    wms = sorted((ROOT / "data" / "watermarks").glob("*.png"))
    attacks = _attack_instances(ROOT / args.attacks)
    rows = []

    for hp in hosts:
        host = read_color(hp)
        ch = host[:, :, method.config.channel]
        h0 = (ch.shape[0] // 2) * 2
        w0 = (ch.shape[1] // 2) * 2
        for wp in wms:
            wm = prepare_binary_watermark(wp, method.config.watermark_size)
            bits = scrambled_bits_from_watermark(wm, method.config.arnold_iterations)
            rr, cc = selected_block_arrays((h0, w0), 2, bits.size, key)

            emb = method.embed(host, wm, key=key)
            fixed = host.copy()
            fch = fixed[:, :, method.config.channel]
            for i in range(bits.size):
                r = int(rr[i]); c = int(cc[i])
                y0, y1 = project_safe_block_q18(
                    int(ch[r, c]), int(ch[r + 1, c]),
                    int(ch[r, c + 1]), int(ch[r + 1, c + 1]),
                    int(bits[i]), DEFAULT_FPGA_FRAC_BITS,
                )
                fch[r, c + 1] = y0
                fch[r + 1, c + 1] = y1

            image_mismatch = int(np.count_nonzero(emb.image != fixed))
            attack_bit_mismatch = 0
            attack_cases = 0
            for attack_name, params in attacks:
                attacked = apply_attack(attack_name, emb.image, **params)
                ref = method.extract(attacked, key=key, side_info=None, watermark_shape=wm.shape).watermark
                qbits = selected_r12_q18_bits(
                    attacked[:, :, method.config.channel], rr, cc, DEFAULT_FPGA_FRAC_BITS
                )
                got = watermark_from_scrambled_bits(
                    qbits, method.config.watermark_size, method.config.arnold_iterations
                )
                attack_bit_mismatch += int(np.count_nonzero(ref != got))
                attack_cases += 1

            rows.append({
                "host": hp.stem,
                "watermark": wp.stem,
                "q18_fraction_bits": DEFAULT_FPGA_FRAC_BITS,
                "watermarked_image_mismatch_samples": image_mismatch,
                "standard_attack_cases": attack_cases,
                "decoder_bit_mismatches": attack_bit_mismatch,
                "psnr_db": float(emb.metadata["actual_psnr_db"]),
                "clean_exact": bool(np.array_equal(method.extract(emb.image, key=key, side_info=None, watermark_shape=wm.shape).watermark, wm)),
            })
            print(
                f"{hp.stem}/{wp.stem}: image_mismatch={image_mismatch}, "
                f"attack_decode_mismatch={attack_bit_mismatch}"
            )

    df = pd.DataFrame(rows)
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print("--- aggregate ---")
    print(f"pairs={len(df)}")
    print(f"watermarked image mismatches={int(df.watermarked_image_mismatch_samples.sum())}")
    print(f"decoder mismatches over standard attacks={int(df.decoder_bit_mismatches.sum())}")
    print(f"clean exact={bool(df.clean_exact.all())}")
    if int(df.watermarked_image_mismatch_samples.sum()) or int(df.decoder_bit_mismatches.sum()):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
