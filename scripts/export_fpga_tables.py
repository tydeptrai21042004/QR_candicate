from __future__ import annotations

"""Export the key/geometry control plane for a streaming FPGA implementation.

No watermarking rule is changed.  The HMAC block order and Arnold transforms
are computed once on the host/control plane and emitted as ROM/BRAM tables so
they never consume the per-frame datapath budget.
"""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qrwatermark.utils.permutation import selected_block_arrays
from qrwatermark.utils.watermark import arnold_destination_map


def _write_hex(path: Path, values: np.ndarray, width_bits: int) -> None:
    digits = (int(width_bits) + 3) // 4
    mask = (1 << int(width_bits)) - 1
    with path.open("w", encoding="ascii") as f:
        for value in np.asarray(values).ravel():
            f.write(f"{int(value) & mask:0{digits}X}\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export blind-v3 FPGA ROM/BRAM tables")
    ap.add_argument("--key", default="blind-v3-fpga")
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--watermark-size", type=int, default=64)
    ap.add_argument("--arnold-iterations", type=int, default=10)
    ap.add_argument("--out", default="fpga/generated")
    args = ap.parse_args()

    h = (int(args.height) // 2) * 2
    w = (int(args.width) // 2) * 2
    wm_n = int(args.watermark_size)
    count = wm_n * wm_n
    key = args.key.encode("utf-8")
    rr, cc = selected_block_arrays((h, w), 2, count, key)
    block_cols = w // 2
    block_index = (rr // 2) * block_cols + (cc // 2)
    payload_index = np.arange(count, dtype=np.uint32)

    # Stream row-major through 2x2 blocks.  The schedule contains only selected
    # blocks, so hardware compares the current block counter with one sorted ROM
    # entry and never performs a random frame-buffer lookup.
    order = np.argsort(block_index, kind="stable")
    sorted_block = block_index[order].astype(np.uint32)
    sorted_payload = payload_index[order]
    packed_schedule = sorted_block | (sorted_payload << 16)

    out = ROOT / args.out
    out.mkdir(parents=True, exist_ok=True)
    _write_hex(out / "carrier_schedule_28b.mem", packed_schedule, 28)
    _write_hex(out / "carrier_block_index_16b.mem", sorted_block, 16)
    _write_hex(out / "carrier_payload_index_12b.mem", sorted_payload, 12)

    arnold_fwd = arnold_destination_map(wm_n, int(args.arnold_iterations), False)
    arnold_inv = arnold_destination_map(wm_n, int(args.arnold_iterations), True)
    _write_hex(out / "arnold_embed_dst_12b.mem", arnold_fwd, 12)
    _write_hex(out / "arnold_inverse_dst_12b.mem", arnold_inv, 12)

    meta = {
        "frame": [h, w],
        "block_size": 2,
        "block_grid": [h // 2, w // 2],
        "payload_bits": count,
        "watermark_size": wm_n,
        "arnold_iterations": int(args.arnold_iterations),
        "qim_period": 48,
        "blind_margin_ratio": 0.125,
        "safe_intervals_mod_48": {"bit0": [6, 18], "bit1": [30, 42]},
        "fixed_point_fraction_bits": 18,
        "schedule_word": {
            "bits_15_0": "row-major 2x2 block index",
            "bits_27_16": "scrambled payload bit index",
        },
        "streaming_note": "process blocks row-major; advance schedule pointer only on a block-index match",
    }
    (out / "fpga_tables.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote FPGA tables to {out}")
    print(f"schedule entries={count}; schedule storage={count*28/8/1024:.2f} KiB")


if __name__ == "__main__":
    main()
