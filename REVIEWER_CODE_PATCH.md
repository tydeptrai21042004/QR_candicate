# Green-Quad reviewer experiment patch — 2026-10-11

This patch contains **only added/changed files**. Merge the directories into
an existing checkout (`QR_candicate-main`), retaining the old file layout.
The mathematical Green-Quad production embed/extract functions are unchanged.
No pilots, ECC, temporal recovery, Arnold scrambling, external side information,
or neural weights have been inserted into the online algorithm.

## Quick check

```bash
python -m pip install -e '.[test]'
PYTHONPATH=src python -m pytest -q tests/test_green_quad_min_energy.py tests/test_final_single_proposal.py tests/test_baselines_smoke.py tests/test_paper_baselines.py tests/test_reviewer_suite.py
PYTHONPATH=src python scripts/run_reviewer_suite.py --mode smoke --sections all --output results/reviewer_smoke
```

## Complete evaluation

```bash
PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python scripts/run_reviewer_suite.py --mode full --sections all \
  --hosts-dir data/hosts/classical --output results/reviewer_full
```

Add additional independent BMP/PNG host images to a host directory and use
`--hosts-dir` for statistical power. The six bundled hosts are a proof of
execution, not an adequate independent test population for strong CIs.

The complete suite writes:

- `protocol_manifest.json`: content hashes and deterministic independent
  training/validation/test host partitions. Defaults to a 2/1/3 split for the
  six-host sample. Method is fixed in advance; training hosts are reserved,
  validation hosts select baseline strength, test hosts evaluate performance.
- `baselines/calibration.csv`: validation-only strength calibration. Frozen
  baseline parameters are applied to held-out test images; no test BER is used
  to select strengths.
- `baselines/psnr_audit.csv`: delta to Green-Quad PSNR, with an explicit
  `within_tolerance` boolean (default 0.2 dB). A mismatch is reported as a
  mismatch, not a fair win. All methods use an identical 4096-bit payload.
- `baselines/operating_point_psnr_calibrated/`: **secondary** study calibrating
  PSNR on each test host while NEVER accessing attacked BER; separately labeled,
  so this does not masquerade as validation-frozen tuning. Runs in FULL mode.
- `baselines/per_attack_ci.csv`, `baselines/paired_host_ci.csv`: host-clustered
  paired 95% bootstrap intervals. Never bootstrap attack rows as independent.
- `security/raw.csv`, `roc.csv`, `report.json`: unmarked negatives, wrong keys,
  one-bit key flips, unrelated templates; validation-negative ROC threshold and
  held-out FPR/TPR. A blind extractor that always returns 4096 bits is NOT
  automatically an ownership detector; a candidate template must be supplied.
- `attacks/raw.csv`, `per_attack_ci.csv`: more JPEG/noise strengths,
  unregistered off-grid rotation, fractional translation, shear, aspect-ratio
  distortion, and a public carrier-null attack. There is NO inverse warp or
  implicit synchronization. Each random seed retains its own row.
- `optimality/integer_projection.csv`: fixed-centre exact integer projection,
  greedy control, neighboring QIM centre energy, rejection/boundary statistics,
  plus fully saturated synthetic hosts. A nearby-centre improvement does NOT
  imply the production algorithm proves global optimality across lattice points.
- `capacity/capacity.csv`, `*.png`: payload size 16²...256² bits, QIM periods
  48...82, PSNR/SSIM, BER/NC, runtime. The 65,536-bit capacity applies only to
  512x512 hosts; watermark content is resized at each evaluated size.
- `runtime/runtime.csv`, `environment.json`: per-host cold-independent warmed
  embedding and extraction latency, median/p95/p99 and same-hardware FPS.
  Codec and offline security/attack analysis are not part of online throughput.

### Neural-model baseline — optional, requires a real pretrained checkpoint

`run_external_neural_baseline.py` accepts a verifiable installed model runner
implementing the embed/extract CLI interface documented in its module header.
It rejects absent weights and mismatched native 4096-bit payload capacity.
No pretrained checkpoint is bundled and **no neural result is claimed until a
real model is installed and run**. Models with smaller payloads require a
separate capacity-controlled table; a 32-bit model is not a 4096-bit peer.
The direct scalar QIM comparison is a reproducible algorithmic **control**,
not a fictional published paper or a substitute for a neural SOTA baseline.

### Existing test-suite limitation

The archived repository still includes 12 obsolete prototype tests importing
removed APIs (`NLMConfig`, `ConvolutionCertifiedR12QIM`, `keyed_blocks_v6`).
Unrestricted `pytest` therefore fails collection *before this patch*. The
Kaggle entry point now runs only the maintained Green-Quad/paper-baseline tests
and the new reviewer tests. Do not report full legacy-suite success.

### Kaggle

After updating GitHub, run `kaggle/QR_KAGGLE_GITHUB_ONE_CELL.py` as a Python
cell; set `QR_SMOKE_ONLY=1` to smoke test or `QR_REVIEW_SUITE=0` to skip reviewer
evaluation. With its default `QR_REVIEW_SUITE=1` it runs the reviewer suite,
keeps periodic ZIP checkpoints, and saves all completed result CSV files. For
actual wall-clock timing use the same notebook hardware for all methods.
