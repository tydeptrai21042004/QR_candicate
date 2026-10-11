from __future__ import annotations

import argparse
from pathlib import Path

from _common import ROOT, resolve
from qrwatermark.core.factory import build_method
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
        if args.mode == "roundtrip":
            ext = method.extract(emb.image, key=key, side_info=emb.side_info, watermark_shape=wm.shape)
            ext_path = out.with_name(out.stem + "_extracted.png")
            write_image(ext_path, ext.watermark)
            print(f"PSNR={psnr(host, emb.image):.4f} SSIM={ssim(host, emb.image):.6f} NC={nc(wm, ext.watermark):.6f} BER={ber(wm, ext.watermark):.6f}")
            print(f"Extracted: {ext_path}")
        print(f"Watermarked: {out}")
        print("Side info: none (fully blind)")
        return

    if not args.image:
        ap.error("extract requires --image; --watermark is optional and used only to print NC/BER")
    image = read_color(resolve(args.image))
    wm = prepare_binary_watermark(resolve(args.watermark), 64) if args.watermark else None
    shape = wm.shape if wm is not None else (int(method.config.watermark_size), int(method.config.watermark_size))
    ext = method.extract(image, key=key,  watermark_shape=shape)
    write_image(args.out, ext.watermark)
    print(f"Extracted: {args.out}")
    if wm is not None:
        print(f"NC={nc(wm, ext.watermark):.6f} BER={ber(wm, ext.watermark):.6f}")
    else:
        print("Fully blind extraction: no side information or reference watermark.")


if __name__ == "__main__":
    main()
