"""Reproducible controlled comparison. Requires PYTHONPATH=src."""
from __future__ import annotations

import argparse
import gc
import json
import statistics
import time
import tracemalloc

import cv2
from pathlib import Path

from qrwatermark.core.config import ProposedConfig
from qrwatermark.proposed import ConvolutionCertifiedR12QIM
from qrwatermark.evaluation.metrics import ber, psnr
from qrwatermark.utils.watermark import prepare_binary_watermark


def run(args):
    x = cv2.imread(str(args.host), cv2.IMREAD_COLOR)
    if x is None:
        raise FileNotFoundError(args.host)
    wm = prepare_binary_watermark(args.watermark, size=64)
    key = b'benchmark-convolution-qr-2026'
    report = {}
    for name, cfg in {
        'v4': ProposedConfig(design='blind_v4', compute_certificate=False),
        'convqr_v5_experimental': ProposedConfig(
            design='convqr_v5', convqr_qr_period=args.period,
            convqr_conv_period=args.period, compute_certificate=False),
    }.items():
        method = ConvolutionCertifiedR12QIM(cfg)
        embedded = method.embed(x, wm, key=key)
        y = embedded.image
        decoded = method.extract(y, key=key)
        attacked = {
            'gaussian3_sigma0.5': cv2.GaussianBlur(y, (3, 3), 0.5),
            'gaussian3_sigma1.0': cv2.GaussianBlur(y, (3, 3), 1.0),
            'jpeg90': cv2.imdecode(
                cv2.imencode('.jpg', y, [cv2.IMWRITE_JPEG_QUALITY, 90])[1], 1),
        }
        results = {
            'psnr_db': psnr(x, y),
            'clean_ber': ber(wm, decoded.watermark),
            'attack_ber': {case: ber(wm, method.extract(image, key=key).watermark)
                           for case, image in attacked.items()},
        }
        for op, fn in {
            'embed': lambda: method.embed(x, wm, key=key),
            'extract': lambda: method.extract(y, key=key),
        }.items():
            # Warm execution (not cold keyed permutation initialization).
            for _ in range(3):
                fn()
            times = []
            for _ in range(args.repeats):
                t0 = time.perf_counter()
                fn()
                times.append(1000 * (time.perf_counter() - t0))
            gc.collect()
            tracemalloc.start()
            fn()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            results[op] = {
                'median_ms': statistics.median(times),
                'max_ms': max(times),
                'peak_python_allocation_mib': peak / 1024**2,
            }
        report[name] = results
    print(json.dumps(report, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=Path, default=Path('data/hosts/classical/airplane.bmp'))
    parser.add_argument('--watermark', type=Path, default=Path('data/watermarks/watermark_1.png'))
    parser.add_argument('--period', type=float, default=24.0)
    parser.add_argument('--repeats', type=int, default=40)
    parser.add_argument('--output', type=Path)
    run(parser.parse_args())
