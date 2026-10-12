# Final code-only integration — 12 October 2026

## Base repository

Apply the files in this ZIP **over the matching paths** in the newest repository
`QR_candicate-main(20261012-153528).zip`. The previous October 12 repository
`QR_candicate-main(20261012-145734).zip` is byte-identical in content.

**Do not delete the repository's existing files.** This archive contains only
modified source/configuration/tests and one optional Kaggle runner, not a copy of
the entire dataset or repository. Do not reapply the two older patches afterward:
all their *relevant source changes* are already integrated here.

## Integrated changes

1. **MMWatermark official-source adapter** — correct RGB `uint8` NumPy input,
   scikit-image `multichannel`/`channel_axis` compatibility, NumPy 2 `float_`
   compatibility, optional automatically cloned official `MM_Robustness`
   companion for attacks requiring it. Attack bodies are not replaced, and
   skipped/failed attacks still prevent strict PASS.
2. **Descriptive hyperparameters** — all six existing hosts × both watermark PNGs
   in full mode; summary has no BER/NC/PSNR/FPS pass/fail criteria and writes no
   `hyperparameter_analysis/acceptance.json`. Period 66 is unchanged. Old
   acceptance JSON is removed when reusing the same hyperparameter output folder.
3. **Separate ablation** — also uses all six hosts and both watermarks in full
   mode. Ablation-specific gates are preserved; those are not hyperparameter gates.
4. **Baseline comparisons at original operating points** — both paper comparison
   YAMLs set `match_psnr_to_proposed: false`. PSNR, BER and NC remain reported;
   methods need not meet a fixed PSNR-tolerance test. Reviewer matched calibration
   is opt-in via `--include-matched-baselines` and is not a default required job.
   Optional matched diagnostic output can be enabled in a YAML using
   `produce_matched_diagnostic: true`. Reporting unequal distortion is essential.
5. **Video evaluation** — run **two watermarks × two codecs** (FFV1 and MP4V),
   separate video and CSV/JSON result folders for all four jobs. Record combined
   embedding + extraction FPS independent of encoding/quality-measurement times.
   Also record a codec-inclusive *stage-based estimate*, clearly distinct from
   end-to-end throughput, which still contains quality-evaluation overhead.
6. **Existing NC=0.90 target** — no such requirement is imposed by these modified
   scripts; mean NC and BER are observed measurements, not a forced target.

## Tests and limits

- **185 targeted automated tests passed** on the integrated code (maintained
  algorithm, main comparison, review suite, orchestrator, MMWatermark fixture,
  launcher, and video tests).
- Hyperparameter `--mode smoke` completed, and `analysis_summary.json` had
  `acceptance_thresholds: null`, `passed_or_failed: null`, no `acceptance.json`.
- Four 12-frame local runs using a short test video constructed from the six
  repository host images completed: FFV1 watermark 1 and 2 BER=0, MP4V both
  returned nonzero BER. Algorithm-only embed+extract above 30 FPS for each.
- Strict upstream **MMWatermark 110/110 attacks was not rerun** locally because
  network-dependent official repository checkout was unavailable. Run on Kaggle
  to establish coverage, not just import compatibility. Full 300-frame four-way
  video results likewise require the Kaggle run.
- The six-host all-samples descriptive analysis is **not an independent held-out
  estimate**. Security threshold calibration still uses separated hosts;
  removing that split without changing the claim would invalidate calibrated
  false-positive statistics. The expensive matched reviewer baseline job is
  optional and disabled by default.
- Unmatched baselines can be compared at their operating points but the study
  cannot claim equal-distortion superiority.

## Run on Kaggle

1. Overlay the supplied source/config/test paths onto your repository, push to
   GitHub `main`.
2. Enable Kaggle Internet and execute the contents of `KAGGLE_FINAL_RUN.py`
   in one Python notebook cell. This runs the merged targeted tests and both
   watermarks under official-source attacks plus FFV1 and MP4V video.
3. Set `RUN_FULL_EXPERIMENT_SUITE=True` in the cell to additionally run the
   computationally expensive main suite. Its matched-baseline calibration is
   optional, not a hidden prerequisite.

For direct local invocation:

```bash
python scripts/run_green_quad_hyperparameters.py --mode full --partition all --min-hosts 6 --repeats 250 --output results/hyperparameters_all6
python scripts/run_green_quad_ablation.py --mode full --partition all --min-hosts 6 --repeats 250 --include-greedy-control --output results/ablation_all6
python scripts/run_existing_hosts_mm_video.py --output results/MM_VIDEO_2WM --min-images 6 --max-images 6 --strict-mm --video-frames 300
python scripts/run_qr_complete_suite.py --mode full --only all --out-root results --max-hours 10.5
# Only if independently desired: add --include-matched-baselines
```

The original `src/qrwatermark/` embedding/extraction implementation, both PNG
watermarks, and six hosts are completely unchanged in this patch.
