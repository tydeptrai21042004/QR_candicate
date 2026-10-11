# Reviewer-audit v2: exact evaluation fixes, not a change to the watermark

This patch leaves Green-Quad T=66 embedding and extraction unchanged. It fixes
four *evaluation* problems; it does not manufacture stronger robustness numbers.

## 1. PSNR matching

`run_reviewer_suite.py` calibrates **one strength scale per baseline**, using
**validation images only**, maximizing the count within ±0.2 dB and then
minimizing the mean absolute PSNR gap. Strength is *frozen* on held-out hosts.
The `baselines/calibration.csv`, `psnr_audit.csv`, `psnr_coverage.csv` and
`metrics_with_eligibility.csv` provide an auditable record. If a paper method
has an irreducible distortion floor or its parameter is discontinuous, exact
matching may be impossible. Such observations are excluded from matched
comparisons but retained in unrestricted descriptive results.

`run_main_comparison.py` is a **secondary operating-point** comparison which
calibrates the embedding PSNR separately for every test image. It is **not** a
locked-validation/generalization experiment; the output now labels this.

## 2. Fair statistical comparison

`paired_host_ci(...,pairing='intersection')` matches each eligible baseline
sample to the identical Green-Quad evaluation condition (host, watermark,
attack, parameters, random seed). It does not demand that unmatched Green-Quad
rows be deleted. The full/unrestricted comparison remains strictly paired.
The matched report records pairing coverage and independent host count. Less
than two matched hosts ⇒ `ci_valid=false` and no numeric confidence interval.
Methods with zero eligible matched samples are explicitly listed as unmatched.
Host-level resampling prevents attacks and watermark keys from being treated
as independent hosts.

## 3. Twelve independent hosts

The original repository includes only **six** host images. This patch cannot
create six additional *independent* photographs by augmentation or duplication.
Full mode now requires **12 real, distinct 512×512 RGB/color images** by default.
Duplicate file content and decoded-pixel content are rejected. If the external
photographs vary in aspect ratio or resolution, stage them with:

```bash
python scripts/prepare_reviewer_hosts.py \
  --input data/hosts/classical \
  --input /kaggle/input/additional-real-host-images \
  --output /kaggle/working/qr_hosts_12
```

The script aspect-preservingly resizes then center-crops to 512×512, saves
original source SHA256 hashes, and flags perceptual near-duplicates for manual
review. It does **not** synthesize new hosts. Supply **at least six genuine new
source photographs depicting different scenes**. Review their provenance, near-
duplicate flags, and any training-data overlap yourself.

```bash
PYTHONPATH=src python scripts/run_reviewer_suite.py \
  --mode full --sections security baselines \
  --hosts-dir /kaggle/working/qr_hosts_12 \
  --security-wrong-keys 64 \
  --output /kaggle/working/QR_REVIEWER_AUDIT_V2
```

Host allocation is deterministic at the **host level** with no split overlap.
For a quick smoke check with the old six images, use `--mode smoke` (not proof of
12-host generalization). To run the full *exploratory* suite on six old images,
explicitly pass `--min-hosts 6` and report it as exploratory.

## 4. Security / ROC

The revised test adds authentic clean and **attacked** positives; wrong keys,
flipped key, wrong watermark, and unmarked negatives under identical attack
settings; reproducible attacks; calibration thresholds on validation hosts;
and locked evaluation on held-out hosts. Defaults to 64 wrong keys per
host/watermark/attack (not 64 independent images).

Outputs: `security/raw.csv`, `roc.csv`, `report.json`, and
`acceptance_by_attack_case.csv`. `report.json` includes host-clustered AUC CIs,
negative sample counts, empirical false positives and an **optimistic IID-only**
Clopper–Pearson bound, alongside a host-level any-false-positive upper bound.
A high AUC or zero wrong-key errors on a few independent hosts **cannot**
certify a population FPR below 1%. With 6 independent held-out hosts, a zero
host-level false-positive observation has a very large upper bound.

The evaluation uses *no* pilots, ECC, repetition, attack-specific branches,
learned networks, or side information. It does not claim to repair the known
geometric and noise robustness limitations.

## Standard comparison with twelve images

```bash
PYTHONPATH=src python scripts/run_main_comparison.py \
  --config configs/experiments/main_comparison.yaml \
  --key reviewer-fixed-key --hosts-dir /kaggle/working/qr_hosts_12 \
  --run-dir /kaggle/working/QR_STANDARD_V2
```

`--min-hosts 6` allows clearly labeled *exploratory* runs on the six originals.
Test run completed on the provided six-image corpus in smoke mode. Full
12-host results require user-supplied independent images; they are not present
in the uploaded repository.

## One Kaggle cell

The new `kaggle/QR_REVIEWER_AUDIT_V2_ONE_CELL.py` is an executable **one-cell
notebook script**. Paste its contents into one Kaggle cell, point
`EXTRA_REAL_HOSTS` at a dataset with at least six additional independent images,
and turn Internet on. It pulls your latest GitHub `main`, prepares the genuine
12-host corpus, runs smoke/regression tests and the *full security and matched
baseline evaluations*, and checkpoints a ZIP every three minutes. The
`security` and `baselines` sections are sequential, not fabricated or parallel
comparisons. Commit the changed patch files to GitHub **before running**.

The script is focused on the four audit gaps. It does **not** rerun the full
runtime/capacity/optimality suites or the original standard paper comparison;
run the separate main-comparison command for those if needed.
