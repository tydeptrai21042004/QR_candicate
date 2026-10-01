from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from _common import ROOT, resolve
from qrwatermark.core.factory import build_method
from qrwatermark.utils.image_io import read_color
from qrwatermark.utils.permutation import clear_permutation_cache
from qrwatermark.utils.watermark import prepare_binary_watermark


def main():
    ap=argparse.ArgumentParser(description="Warm-session edge latency benchmark for CCQR v5 runtime path")
    ap.add_argument("--host",default="data/hosts/classical/girl.bmp")
    ap.add_argument("--watermark",default="data/watermarks/watermark_1.png")
    ap.add_argument("--config",default="configs/methods/proposed.yaml")
    ap.add_argument("--key",default="ccqr-edge-runtime-2026")
    ap.add_argument("--repeat",type=int,default=10)
    ap.add_argument("--target-fps",type=float,default=30.0)
    args=ap.parse_args()

    method=build_method("proposed",resolve(args.config))
    host=read_color(resolve(args.host))
    wm=prepare_binary_watermark(resolve(args.watermark),method.config.watermark_size)
    key=args.key.encode()

    # Warm/cache session control-plane descriptors.
    first=method.embed(host,wm,key=key)
    method.extract(first.image,key=key,side_info=first.side_info,watermark_shape=wm.shape)

    et=[]; xt=[]
    for _ in range(max(1,args.repeat)):
        t=time.perf_counter(); emb=method.embed(host,wm,key=key); et.append(time.perf_counter()-t)
        t=time.perf_counter(); method.extract(emb.image,key=key,side_info=emb.side_info,watermark_shape=wm.shape); xt.append(time.perf_counter()-t)

    target_ms=1000.0/args.target_fps
    def report(name,times):
        a=np.asarray(times)*1000.0
        med=float(np.median(a)); p95=float(np.quantile(a,.95)); fps=1000.0/med
        print(f"{name}: median={med:.3f} ms p95={p95:.3f} ms ~= {fps:.2f} fps; {args.target_fps:g}fps={'PASS' if med<=target_ms else 'NOT YET'}")
    report("certified_embed_python",et)
    report("extract_python_warm",xt)

    h,w=host.shape[:2]
    scans=1+int(method.config.certificate_max_passes)  # base frame + max certificate frame scans
    mpix=float(h*w*args.target_fps*scans)/1e6
    print(f"hardware_cycle_budget: {h}x{w}, {args.target_fps:g} fps, {scans} frame traversals -> {mpix:.3f} Mpixel/s")
    print("Assumes the two 3x3 extreme convolution kernels operate in parallel and final path tightening reuses the last extreme-filter outputs.")


if __name__=="__main__":
    main()
