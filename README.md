# QR Candidate — CCQR-R12-QIM research code

This repository implements the revised proposal method:

**CCQR-R12-QIM: Convolution-Certified R12-QIM Watermarking with Repeated Soft Recovery.**

The proposal no longer uses adaptive Q/R branch selection.  The core method uses only the canonical QR statistic `R[0,1]` (`r12`), a finite-torus convolution model, payload-independent convolution-certified QIM-period selection, range-feasible lattice projection, and repeated soft decoding.

## Main proposal changes

- **R12-only QR embedding.** The proposal embeds a binary watermark in the off-diagonal canonical QR coefficient `r12`; the Q branch and old `r22` branch are not part of the proposal path.
- **Exact local distortion model.** If `delta` is the change in `r12`, orthogonal invariance gives exact continuous 2x2-block MSE `delta^2 / 4`.
- **Finite-torus convolution model.** Normalized Gaussian kernels are represented as circular convolution operators on `Z_M x Z_N`; the implementation also records `max |H(omega)-1| = ||T_h-I||_(2->2)`.
- **Payload-independent period selection.** Before the actual watermark bit is read, each repeated block group is tested against the configured convolution family.  The encoder chooses among a discrete period table using the worst observed `r12` convolution shift, an explicit safety factor, additive/rounding budgets, and a distortion constraint.
- **Range-feasible QIM projection.** Among lattice points encoding the same bit, the encoder selects the nearest point whose reconstructed block remains inside `[0,255]`.  This avoids clipping invalidating the QR distortion model.
- **Compact authenticated side information.** Only the period-table index is stored per payload bit.  Three periods require two authenticated side-information bits per watermark bit.
- **Repeated soft recovery.** `r` repeated observations are accumulated as signed QIM evidence; ground truth is never used by the decoder.
- **No heuristic Q/R score.** The previous `robustness/(MSE+eps)` and four-perturbation branch score are not used by the proposal.

## Default proposal configuration

```yaml
repetition: 5
period_candidates: [48.0, 52.0, 56.0]
max_group_mse: 220.0
convolution_kernel_size: 3
convolution_gaussian_sigmas: [0.50, 0.75]
convolution_safety_factor: 1.10
additive_feature_budget: 4.0
rounding_feature_budget: 1.0
```

NLM is disabled by default.  It remains available only as an ablation option.

## Relevant source files

```text
src/qrwatermark/proposed/
├── convolution.py     # finite-torus convolution + spectral operator severity
├── certificate.py     # payload-independent certificate / period optimization
├── r_branch.py        # canonical R12-QIM + range-feasible lattice projection
├── embedding.py
├── extraction.py
├── soft_decoder.py
├── side_info.py
└── method.py
```

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

## Main experiments

```bash
python scripts/run_main_comparison.py \
  --config configs/experiments/main_comparison.yaml \
  --key "replace-with-your-experiment-key" \
  --run-dir runs/main_comparison

python scripts/run_ablation.py --key "..." --run-dir runs/ablation
python scripts/run_leakage_test.py --key "..." --run-dir runs/leakage
python scripts/run_repetition_study.py --key "..." --run-dir runs/repetition
```

The leakage experiment now tests mutual information between the **payload-independent period code** and the watermark bits; there are no Q/R mode flags in CCQR-R12-QIM.

## Tests

```bash
pytest -q
```

The tests include canonical QR reconstruction, exact R12 distortion, convolution-operator checks, payload-independent certificate selection, side-information authentication, clean end-to-end recovery, attacks, metrics, and baseline smoke tests.

## Baseline status

The literature baselines remain under `src/qrwatermark/baselines/` and use the same benchmark/evaluation pipeline.  Their implementation status is documented in each baseline directory.

## Evaluation rule

Ground truth is used only after extraction to calculate metrics such as BER and NC.  It is never used to choose a period, invert a watermark, select a preprocessing candidate, or otherwise modify the decoder result.
