# Integer Convolution Carrier — Tested Source Patch

**Recommended experimental setting: Quad-v9, period 70.**

This ZIP contains only changed/new files, not the full QR_Candidate repository. It was developed against the **V5 full project with the tested FCQR-v6 patch and the V6 speed patch applied**. Existing V5 and V6 algorithms remain available. It also includes the earlier 8-pixel (v7) and paired 8-pixel (v8) variants for a reproducible ablation.

## Install

1. Start with your `QR_candicate-main` directory containing the FCQR-v6 code.
2. Back up `src/qrwatermark/core/config.py`, `src/qrwatermark/core/factory.py`, and `src/qrwatermark/proposed/method.py` if you have customized them or installed additional methods such as PCQR. These are V6-base replacements, not a universal merge against every third-party patch.
3. Extract this ZIP **inside** the repository root. Keep directory structure.
4. Run:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
PYTHONPATH=src python scripts/benchmark_integer_carriers.py --out evidence_integer_methods
```

### Try the recommended method

```python
import cv2
from qrwatermark.core.factory import build_method
from qrwatermark.utils.watermark import prepare_binary_watermark

method = build_method('proposed', 'configs/methods/proposed_integer_conv_quad_v9_experimental.yaml')
host = cv2.imread('data/hosts/classical/airplane.bmp')
watermark = prepare_binary_watermark('data/watermarks/watermark_1.png', 64)
embedded = method.embed(host, watermark, key=b'my-own-key')
assert embedded.side_info is None
extracted = method.extract(embedded.image, key=b'my-own-key').watermark
```

## Mathematical construction

The 4×4 host patch is divided into four disjoint 2×2 convolution supports. Each cell carries one bit. With cell samples `x_1,...,x_4` and fixed normalized direction `q=(1,1,1,1)/2`, its QR cross coefficient is `r12=qᵀx=S/2`, where `S=sum(x_i)`. A QR algorithm does *not* need to run: an **integer QIM** embeds one bit in `S`, with a fixed period `T=70` and centers at 17 (bit zero) and 52 (bit one), modulo 70.

For integer target `S*`, the embedding map writes `S*-S=4q+r`, `0<=r<4`, and adds `q+1` to `r` pixel coordinates and `q` to the rest. Saturated coordinates use a bounded deterministic redistribution that closes the exact integer sum. Extraction is always the same rule `bit = [(S mod 70) >= 35]`. This procedure requires no attack classifier, optimization solver, reference image, auxiliary bits, ECC, duplicated watermark bits, or side information.

Four bits share one keyed 4×4 block lookup. Each bit still has its own **disjoint low-pass support**; the method does not share a noise-sensitive QR anchor.

**Novelty caveat:** this is a fixed-direction low-pass QR *interpretation*, not a fundamentally new general-purpose matrix decomposition. The efficiency comes from exploiting the constrained algebraic structure and reducing independent block mappings.

## Reproduced metrics (same benchmark session)

Source: `evidence/summary.json` and `evidence/per_image_25_attacks.csv`. Six 512×512 BMP hosts × two 64×64 watermarks × 25 attacked configurations = 300 attacked evaluations per method. All use the same fixed green channel and key; measurements are exploratory/in-sample.

| Method | Mean attacked NC | Minimum clean PSNR | Max clean BER | FPS (embed+extract) | Peak Python embed allocations |
|---|---:|---:|---:|---:|---:|
| FCQR-v6 | 0.775643 | 50.1089 dB | 0 | 266.51 | 1.546 MiB |
| Integer Conv8-v7 | 0.777745 | 50.0394 dB | 0 | 229.99 | 1.040 MiB |
| Integer ConvPair-v8 | 0.777864 | 49.9536 dB | 0 | 277.83 | 0.947 MiB |
| **Integer ConvQuad-v9, T=70** | **0.780126** | **50.0377 dB** | **0** | **369.49** | **0.868 MiB** |

On this benchmark, Quad-v9 is ~1.39× as fast as V6 and reduces peak Python embedding allocations by ~44%. `tracemalloc` allocation peaks are **not total RSS RAM**. Throughput depends on CPU, execution environment, and implementation. This is not a demonstrated ≥2× gain, and the NC/PSNR constraints are exceeded by only narrow margins.

Regression test evidence: `evidence/pytest_all.log` (**644 passed**). Parameters were selected using this evaluation set; holdout evaluation on genuinely unseen images and independent keys is essential before publication claims. Geometric attacks still require attention.

## Files

- `src/qrwatermark/proposed/integer_convolution_quad_v9.py`: preferred four-carrier algebraic method
- `src/qrwatermark/proposed/integer_convolution_v7.py`: exact integer QIM helper and one-carrier reference
- `src/qrwatermark/proposed/integer_convolution_pair_v8.py`: two-carrier ablation
- `src/qrwatermark/core/{config.py,factory.py}` and `src/qrwatermark/proposed/method.py`: method registration
- `configs/methods/`: experiment configurations
- `tests/`: new image-level and arithmetic regression tests
- `scripts/benchmark_integer_carriers.py`: reproducible comparison using `benchmark_fcqr_v6.py` attack suite
- `evidence/`: CSV, JSON, test log and preview
