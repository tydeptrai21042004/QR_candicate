# QR Candidate — Fully Blind QR-QIM research code

The default proposal is now **Blind ME-CQR-QIM v3** (`blind_mecqr_qim_v3`). It is a fully blind QR watermarking method: extraction needs only the received image, the secret key, and the public method configuration. No original host, selector map, period-code stream, certificate mask, reference watermark, or `.side.npz` file is required.

The older selector-based **ME-CQR-QIM v2** and repeated **CCQR-R12-QIM v1** are preserved as historical ablations so earlier results remain reproducible.

## Current proposal path

For each payload bit, the key deterministically assigns exactly one non-overlapping 2x2 carrier. Writing the carrier as `A=[a,b]`, the method uses

```text
q1  = a / ||a||
r12 = q1^T b
b'  = b + delta q1
```

so changing `r12` by `delta` costs exactly `delta^2` continuous embedding energy. With period `Delta`, the default safe decision sets are

```text
S0 = union_k Delta [k+1/8, k+3/8]
S1 = union_k Delta [k+5/8, k+7/8]
```

and the embedder solves one scalar Euclidean projection per bit:

```text
min delta^2
subject to r12 + delta in S_bit
           0 <= b + delta q1 <= 255
```

The decoder regenerates the same one-to-one carrier map from the key and applies only the hard half-period decision. There is **no carrier selector**, no per-bit adaptive period, no repetition, no majority vote, and no decoder oracle.

## Fully blind contract

- payload copies per bit: **1**
- selector bits: **0**
- period-code bits: **0**
- certificate-mask bits: **0**
- side-info authentication bits: **0**
- serialized side information: **0 bits**
- original host required for extraction: **no**
- reference watermark required for extraction: **no**

The optional convolution certificate is a report-only post-embedding audit. It never modifies pixels and is never serialized or consumed by extraction.

## Default configuration

```yaml
design: blind_v3
block_size: 2
channel: 0
watermark_size: 64
arnold_iterations: 10
qim_period: 48.0
blind_margin_ratio: 0.125
blind_projection: safe_set
compute_certificate: true
store_certificate_mask: false
```

The default safe margin is `Delta/8 = 6`. Since only the two pixels of the second QR column are rounded, the induced `r12` rounding error is bounded by `sqrt(2)/2`, so the default clean decision has a strict rounding margin.

## Main source files

```text
src/qrwatermark/proposed/
├── blind_embedding.py    # fully blind safe-set projection + report-only certificate
├── blind_extraction.py   # image + key only decoder
├── elegant_embedding.py  # historical selector-based v2
├── elegant_extraction.py
├── embedding.py          # historical repeated v1
├── extraction.py
├── qr.py
├── qr_sensitivity.py
└── method.py
```

See `docs/FULLY_BLIND_V3.md` and `RUN_FULLY_BLIND_V3.md` for the derivation and commands.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install -e .
```

## One-image round trip

```bash
python scripts/run_method.py roundtrip \
  --method proposed \
  --config configs/methods/proposed.yaml \
  --host data/hosts/classical/girl.bmp \
  --watermark data/watermarks/watermark_1.png \
  --key "replace-with-your-experiment-key" \
  --out runs/example/watermarked.png
```

## Fair main comparison

`configs/experiments/main_comparison.yaml` enables `match_psnr_to_proposed: true`. For each host/watermark pair, the proposal is embedded first and each baseline quantization strength is tuned to the proposal's **realized embedding PSNR** before robustness is compared.

```bash
python scripts/run_main_comparison.py \
  --config configs/experiments/main_comparison.yaml \
  --key "replace-with-your-experiment-key" \
  --run-dir runs/main_comparison
```

The benchmark embeds each method **once per host/watermark pair** and reuses that exact watermarked image for every attack and random seed.

## Metrics

The CSV now separates:

- `embedding_psnr`, `embedding_ssim`: original host vs. watermarked image;
- `attacked_psnr`, `attacked_ssim`: original host vs. attacked watermarked image;
- `ber`, `nc`: recovery metrics;
- `side_information_bits`: side-information cost;
- `certified_fraction`: proposal-only certificate coverage.

The legacy `psnr` and `ssim` columns remain aliases of the **embedding** metrics so old plotting scripts do not silently use attack distortion as imperceptibility.

## Attack terminology

- `random_pixel_dropout`: independent random missing pixels (formerly called `occlusion`);
- `rectangular_occlusion`: one contiguous covered rectangle;
- `registered_rotation_resample`: rotate and inverse-rotate, measuring interpolation/resampling damage;
- `rotation_unregistered`: actual unknown rotation/synchronization stress.

Old attack names remain aliases for compatibility but new configs use the precise names.

## Side information

The **default v3 proposal reports exactly zero side-information bits** and `EmbeddingResult.side_info` is `None`. Historical v2/v1 ablations retain their former selector/period metadata only so their earlier experiments remain reproducible. Baselines continue to report their own protocol-specific overhead where applicable.

For the adaptive-QIM control, use the capacity-compatible profile:

```bash
python scripts/run_main_comparison.py \
  --config configs/experiments/qim_control_comparison.yaml \
  --key "replace-with-your-experiment-key" \
  --run-dir runs/qim_control
```

This profile uses a common 16x16 watermark because Zareian-2013 has one native bit per 16x16 host block; forcing the 64x64/4096-bit main payload would exceed the paper method's native capacity on a 512x512 host.

## Tests

```bash
pytest -q
```

In addition to unit tests, the theorem regression is non-vacuous: it requires certified groups and checks that every certified group decodes correctly under every configured extreme convolution kernel and several unseen convex mixtures.

The v4 suite also verifies that the closed-form QR-free reconstruction matches the canonical `Q @ R_new` reference after rounding, that the comparator/triangular QIM decision matches the former sinusoidal hard decision, that the piecewise path theorem bounds dense convex-path samples, and that final certificate tightening leaves embedded pixels and period codes unchanged.

## Hardware-oriented implementation

For `A=[a,b]`, the carrier and reconstruction are implemented directly as

```text
r12 = (a^T b) / ||a||
b'  = b + delta * a / ||a||
```

so a production datapath does not need a matrix QR engine. The decoder likewise uses a phase comparator and triangular confidence instead of a trigonometric soft score. The bounded spread projection already has a closed-form fast path (`delta = d w`) and uses the iterative controller only when a pixel-range bound becomes active.

See `docs/hardware_deployment_v4.md` and run:

```bash
python scripts/run_hardware_sanity.py
```

The control-plane operations (HMAC, keyed permutation, serialization) are intentionally kept separate from the streaming arithmetic datapath.

## Baseline status

`Nha2022ImprovedQR` is implemented from the published rule. `Su2020Schur` preserves the paper's required mode-flag side information, while `Zareian2013AdaptiveQIM` preserves its published entropy-map/parameter side information and rotation-registration stage. These are protocol-faithful reimplementations, not claims of author-source identity.

## Evaluation rule

Ground truth is used only after extraction to calculate BER/NC or in theorem regression tests. It is not used by the production decoder to flip, align, or choose extracted bits.

## v5 real-time edge execution path

The v5 runtime refactor preserves the v4 mathematical method and output while moving session-invariant work out of the frame loop and batching the 2x2/5-carrier arithmetic. Key changes are cached HMAC block descriptors, vectorized `r12` extraction/QIM decoding, a batched four-period candidate bank, vectorized certificate bounds, and reuse of the last extreme-filter blocks for final path tightening.

Use:

```bash
python scripts/run_edge_benchmark.py --repeat 10 --target-fps 30
```

The benchmark reports software embed/extract latency separately and the conservative hardware pixel-rate budget. Do not claim 30-fps certified embedding from Python timings alone; see `docs/edge_realtime_v5.md` for the ARM control-plane + FPGA/SIMD data-plane architecture.
