from __future__ import annotations

import argparse
from pathlib import Path

from _common import ROOT, resolve
from qrwatermark.core.factory import build_method
from qrwatermark.proposed.side_info import load_side_info, save_side_info
from qrwatermark.utils.image_io import read_color, write_image
from qrwatermark.utils.watermark import prepare_binary_watermark
from qrwatermark.evaluation.metrics import ber, nc, psnr, ssim


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["embed", "extract", "roundtrip"])
    ap.add_argument("--method", default="proposed")
    ap.add_argument("--config", default="configs/methods/proposed.yaml")
    ap.add_argument("--host")
    ap.add_argument("--watermark")
    ap.add_argument("--image")
    ap.add_argument("--side-info")
    ap.add_argument("--key", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    method = build_method(args.method, resolve(args.config))
    key = args.key.encode("utf-8")
    if args.mode in {"embed", "roundtrip"}:
        if not args.host or not args.watermark:
            ap.error("embed/roundtrip require --host and --watermark")
        host = read_color(resolve(args.host))
        wm = prepare_binary_watermark(resolve(args.watermark), 64)
        emb = method.embed(host, wm, key=key)
        out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
        write_image(out, emb.image)
        side_path = Path(args.side_info) if args.side_info else out.with_suffix(".side.npz")
        if method.name == "ccqr_r12_qim_v1":
            save_side_info(side_path, emb.side_info)
        if args.mode == "roundtrip":
            ext = method.extract(emb.image, key=key, side_info=emb.side_info, watermark_shape=wm.shape)
            ext_path = out.with_name(out.stem + "_extracted.png")
            write_image(ext_path, ext.watermark)
            print(f"PSNR={psnr(host, emb.image):.4f} SSIM={ssim(host, emb.image):.6f} NC={nc(wm, ext.watermark):.6f} BER={ber(wm, ext.watermark):.6f}")
            print(f"Extracted: {ext_path}")
        print(f"Watermarked: {out}")
        if emb.side_info is not None:
            print(f"Side info: {side_path}")
        return

    if not args.image or not args.watermark:
        ap.error("extract requires --image and --watermark (watermark is used only for shape/output metric, never by the decoder)")
    image = read_color(args.image)
    wm = prepare_binary_watermark(args.watermark, 64)
    side = None
    if args.side_info:
        side = load_side_info(args.side_info) if method.name == "ccqr_r12_qim_v1" else None
    ext = method.extract(image, key=key, side_info=side, watermark_shape=wm.shape)
    write_image(args.out, ext.watermark)
    print(f"Extracted: {args.out}")
    print(f"NC={nc(wm, ext.watermark):.6f} BER={ber(wm, ext.watermark):.6f}")


if __name__ == "__main__":
    main()
