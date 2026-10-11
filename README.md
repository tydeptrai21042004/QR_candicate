# Green-Quad Minimum-Energy QIM — single proposed method

This repository contains **exactly one proposed watermarking method**, `green_quad_min_energy`, configured for **OpenCV BGR green channel (index 1)** and **integer QIM period 66**. Previous V1–V9 experimental proposal implementations have been removed. Published-method baselines in `src/qrwatermark/baselines` are retained for independent comparison and are not proposed methods.

The algorithm uses four disjoint 2×2 integer-sum carriers within a keyed 4×4 block. It embeds one bit per carrier with a quantization lattice and computes the globally minimum squared-pixel-error integer projection **for each chosen target sum**. It requires no host image or extra side information during extraction. No pilot, ECC or repeated embedding is used.

## Run

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python scripts/run_method.py roundtrip --key example --host data/hosts/classical/girl.bmp --watermark data/watermarks/watermark_1.png --out /tmp/stego.png
python scripts/run_green_quad_ablation.py --mode quick --repeats 50 --output results/quick
python scripts/run_green_quad_ablation.py --mode full --repeats 250 --output results/full
```

The ablation runner produces `attack_rows.csv` (all per-attack observations), `by_attack.csv`, `summary.csv`, `runtime.csv`, `multikey_clean.csv`, `environment.json`, and `acceptance.json`. It compares **only internal components of the same algorithm**: periods 64/66/68, RGB channels, Arnold on/off, and optionally `--include-greedy-control` for checking the integer solver's distortion benefit. There is no parameter tuning on evaluation attacks.

The original six hosts and two watermarks are provided; a strict 12-distinct-host study requires adding six additional independent images. Bit-error rate (BER) should be interpreted alongside NC (which can inflate similarity for mostly white binary patterns). The method is not rotation- or crop-invariant.

### Previously observed operating point (NOT a new guarantee)

On the historical six-host benchmark with one key: attacked mean BER ≈ **0.2399**, minimum clean PSNR ≈ **50.45 dB**, clean BER **0**, mean NC ≈ **0.781**, SSIM ≈ **0.99675**. These must be rerun on your target Kaggle hardware. **Combined FPS >175** is a hardware-dependent acceptance criterion; always use `runtime.csv` from the same machine and check p95 latency too. The method does not establish major mathematical novelty on its own.

## Source

- `src/qrwatermark/proposed/green_quad_min_energy.py`: self-contained carrier, lattice, exact integer projection.
- `src/qrwatermark/proposed/method.py`: sole proposal interface.
- `configs/methods/proposed.yaml`: fixed production defaults.
- `scripts/run_green_quad_ablation.py`: full ablation protocol.
- `docs/FINAL_GREEN_QUAD_ONLY.md`: research interpretation, numerical constraints, and limitations.
