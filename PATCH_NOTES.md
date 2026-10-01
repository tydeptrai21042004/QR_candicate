# Corrected paper-baseline patch

This patch is intended to be extracted over the uploaded `QR_candicate-main (1)` repository.
It changes only the files needed to audit/reimplement the paper baselines and comparison harness.

## Main 64x64-payload comparison

`configs/experiments/main_comparison.yaml` contains the proposal plus six decomposition/QR papers:

1. `su2014_qr` — Su et al. (2014), QR `r14`, published candidate quantizer, `Delta=42`.
2. `su2016_hessenberg` — Su (2016), Hessenberg-Q `q22/q32`, `T=0.042`.
3. `su2017_improved_qr` — Su et al. (2017), 3x3 QR `q21/q31`; explicitly paper-aligned reconstruction because author source/full numerical rule is not public here.
4. `su2020_schur` — Su, Zhang & Wang (2020), both U and D candidates, minimum-distortion selection, mode flag, `T=0.03`, `Delta=25`.
5. `chen2021_qqrd` — Chen et al. (2021), quaternion QR carrier, three bits per selected 4x4 quaternion block.
6. `nha2022_improved_qr` — Nha et al. (2022), R-first improved QR and direct `R11` extraction, `q=10`.

## QIM control comparison

`configs/experiments/qim_control_comparison.yaml` adds:

7. `zareian2013_aqim` — Zareian & Tohidypour (2013), high-entropy 16x16 blocks, two-level Haar DWT, adaptive QIM and gain-aware extraction.

This control profile uses a 16x16 (256-bit) common payload because the paper method has one native bit per 16x16 host block and therefore cannot carry the repository's 4096-bit main payload on a 512x512 image without inventing a different method.

## Validation performed

- `pytest -q`: **65 passed**.
- Native 64x64 clean/default-strength audit on Lenna + watermark_1: all six decomposition/QR paper baselines and the proposal achieved clean BER 0 in `validation/main64_clean_default_strength.csv`.
- Capacity-compatible 16x16 PSNR-matched diagnostic includes all seven paper baselines plus proposal and four representative conditions (clean, JPEG-50, Gaussian blur sigma=1, Gaussian noise variance=0.003). See `validation/baseline_audit_comparison.csv` and `validation/baseline_audit_summary.csv`.

The validation CSVs are smoke/audit evidence, not a replacement for the full multi-host, multi-watermark, multi-seed manuscript experiment.

## Run

```bash
pytest -q

python scripts/run_main_comparison.py \
  --config configs/experiments/main_comparison.yaml \
  --key "your-experiment-key" \
  --run-dir runs/main_comparison

python scripts/run_main_comparison.py \
  --config configs/experiments/qim_control_comparison.yaml \
  --key "your-experiment-key" \
  --run-dir runs/qim_control
```
