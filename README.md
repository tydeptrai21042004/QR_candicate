# QR Candidate — corrected research repository

This repository is a notebook-free refactor of the original `QR_candicate` prototype.

## Main corrections

- Removed the monolithic Jupyter workflow.
- Proposed method is a normal Python package under `src/qrwatermark/proposed/`.
- Replaced the old fixed `|q22|-0.5` Q branch with symmetric **Q-angle QIM**.
- Replaced the old diagonal `r22` R branch with **R12 QIM**, avoiding canonical QR sign reversal.
- Q/R branch selection is **payload-independent**: the selector evaluates hypothetical bit 0 and bit 1 before it is allowed to see the real payload bit.
- Replaced `robust_score / (MSE + 1e-9)` with a bounded distortion penalty.
- One Q/R mode flag is stored per payload bit rather than per repeated embedding block.
- Mode flags are bit-packed and HMAC-authenticated.
- Replaced floating-point logistic-map block ordering with HMAC-SHA256 keyed ordering.
- Removed hard-coded `/content` paths and hard-coded private keys.
- Removed ground-truth-assisted watermark inversion/alignment from the evaluator.
- NC no longer compares the extracted watermark with its inverse and silently keeps the larger value.
- Added BER as a first-class metric.
- Baselines and the proposed method share the same attack/evaluation pipeline.
- Added tests for QR reconstruction, both proposal branches, side information, metrics, attacks, baselines, and clean end-to-end extraction.

## Repository layout

```text
QR_candicate_corrected/
├── configs/
│   ├── methods/
│   ├── experiments/
│   └── attacks/
├── data/
│   ├── hosts/classical/
│   └── watermarks/
├── src/qrwatermark/
│   ├── core/
│   ├── proposed/
│   ├── baselines/
│   │   ├── su2017_hessenberg/
│   │   ├── su2020_schur/
│   │   └── nha2022_improved_qr/
│   ├── attacks/
│   ├── evaluation/
│   └── utils/
├── scripts/
└── tests/
```

There is deliberately no `docs/`, `paper/`, `outputs/`, or Jupyter notebook in the repository.
Run directories are created only when you explicitly supply `--run-dir`.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
pip install -e .
```

## One-image proposed-method round trip

```bash
python scripts/run_method.py roundtrip \
  --method proposed \
  --config configs/methods/proposed.yaml \
  --host data/hosts/classical/girl.bmp \
  --watermark data/watermarks/watermark_1.png \
  --key "replace-with-your-experiment-key" \
  --out runs/example/watermarked.png
```

The decoder receives only the watermarked image, key, authenticated side information, and algorithm parameters. The original watermark is never used to alter the decoded result.

## Main comparison

```bash
python scripts/run_main_comparison.py \
  --config configs/experiments/main_comparison.yaml \
  --key "replace-with-your-experiment-key" \
  --run-dir runs/main_comparison
```

## Other experiments

```bash
python scripts/run_robustness.py --key "..." --run-dir runs/robustness
python scripts/run_attack_sweep.py --key "..." --run-dir runs/attack_sweep
python scripts/run_ablation.py --key "..." --run-dir runs/ablation
python scripts/run_leakage_test.py --key "..." --run-dir runs/leakage
python scripts/run_repetition_study.py --key "..." --run-dir runs/repetition
python scripts/run_runtime.py --key "..." --run-dir runs/runtime
```

## Tests

```bash
pytest -q
```

## Baseline status

### Nha et al. 2022

The Nha/Thanh/Phong baseline follows the published 4x4 blue-channel R(1,1) quarter-period QIM embedding equations and the published extraction shortcut `R(1,1) = ||first column||`.

### Su & Chen 2017 and Su et al. 2020

The uploaded original repository contained no baseline source code. Public descriptions expose the principal mechanisms but not all equation-level implementation details needed to claim author-identical code. Therefore the included Su 2017 Hessenberg and Su 2020 Schur modules are **transparent reconstructed reference implementations**, not the authors' official source code. Their directories state this explicitly. If the exact original implementation or complete algorithm equations become available, replace only those baseline modules; the shared benchmark code does not need to change.

## Important evaluation rule

Ground truth is used only after extraction to compute metrics such as BER and NC. It must never be used to choose inversion, orientation, branch, preprocessing candidate, or any other decoder decision.
