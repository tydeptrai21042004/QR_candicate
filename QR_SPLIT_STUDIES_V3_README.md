# QR_candicate V3: separate ablation and hyperparameter analyses

This patch is an **evaluation-only update** based on repository revision
`6191f711ce20260d497d08c40a3204942997f9a1` plus reviewer audit V2.
The proposed embed/extract implementation and final production T=66 configuration
remain unchanged. No ECC, pilots, repetition, synchronization module, or
attack-dependent switching is introduced.

## Study definitions and leakage prevention

| Experiment | Variants | Data partition | Output directory |
|---|---|---|---|
| Ablation / structural controls | final T=66, blue channel, red channel, non-optimal greedy projection | **test** (held-out) | `ablation/` |
| Hyperparameter sensitivity | final T=66, periods T=60,62,64,68,70,72, Arnold iterations 5 and 10 | **validation only** | `hyperparameter_analysis/` |

The final production T=66 is an **identical reference** in both studies.
Ablation and hyperparameter experiments write separate `summary.csv`,
`by_attack.csv`, `attack_rows.csv`, `runtime.csv`, `multikey_clean.csv`,
`acceptance.json`, `environment.json` and `study_protocol.json` files.
Extra outputs: `ablation/ablation_deltas.csv` and
`hyperparameter_analysis/hyperparameter_selection.json`.

Hyperparameter selection minimizes average BER subject to **minimum PSNR > 50 dB**
and **clean BER = 0** on *validation only*. The report identifies the selected
validation candidate but **does not silently replace the final T=66**.
Ablations remain held-out test evaluations; never select a new hyperparameter
using their results. The method's FPS measurements are hardware specific.

The six bundled hosts produce exploratory results only. A defensible
12-host analysis requires **six additional independent genuine images**.
Do not duplicate or augment the existing six to meet that requirement.

## Kaggle, one-cell, all active experiments

1. Merge this patch at your repository root, preserving relative paths. Push to `main`.
2. In Kaggle, enable **Internet**, use a CPU notebook and (strongly recommended)
   attach a dataset with at least six distinct additional real host images.
3. Copy/paste **all** of `kaggle/QR_ALL_RESULTS_SPLIT_V3_ONE_CELL.py` into one cell.
4. Optionally set the `QR_EXTRA_HOSTS_DIR` environment variable to your attached
   image directory; if omitted, the runner labels results exploratory under 12.
5. The default `QR_EXPERIMENT_MODE=full` runs all **eleven** active jobs:
   tests, ablation, hyperparameter analysis, reviewer security, attacks,
   integer optimality, capacity, runtime, 256-bit baseline, standard 4096-bit
   baseline and locked-validation matched baseline.

**Real pretrained neural model comparison is optional** and only runs if an
actual runner and compatible 4096-bit checkpoint are supplied. A missing model
is marked as not supplied, never replaced with synthetic results.

The result archive is `QR_ALL_RESULTS_V3_<commit>_<dataset-hash>_FULL.zip`
inside `/kaggle/working`. It contains **separate study folders** and all
remaining output categories. `job_status.json` records completed, failed,
timed-out and missing jobs, so a partially complete archive is never presented
as fully complete.

The orchestrator creates ZIP checkpoints periodically and after each job.
Archive writes use a temporary file followed by an atomic rename. Job statuses
are not marked complete unless the exit code is zero **and expected files are
present and nonempty**. Runtime results with hidden per-method errors are
rejected even if the wrapper process returns success. Interrupted processes
are terminated as a process group; the final checkpoint preserves data already
written. The 11.58-hour default budget reserves 8 minutes for packaging.
**No script can guarantee that a kernel or machine will not crash, or that all
full experiments will fit Kaggle's session limit.** If a job fails, other jobs
continue; rerunning the cell on a persistent Kaggle working directory resumes
jobs verified as complete for the same exact commit and dataset fingerprint.

### Run only a study

After cloning / installing, or with `PYTHONPATH=src` from the repository root:

```bash
python scripts/run_green_quad_ablation.py --study ablation \
  --mode full --include-greedy-control --repeats 250 \
  --output results/ablation
python scripts/run_green_quad_hyperparameters.py \
  --mode full --repeats 250 --output results/hyperparameter_analysis
```

For prepared external hosts, add `--hosts-dir <prepared_host_dir> --min-hosts 12`
to both commands. To run one suite group through the resumable orchestrator:

```bash
python scripts/run_qr_complete_suite.py --mode full --only ablation
python scripts/run_qr_complete_suite.py --mode full --only hyperparameters
python scripts/run_qr_complete_suite.py --mode full --only all
```

Use `--mode smoke` only for a rapid structural test: it **does not** run the
two full paper-comparison suites. Full is required for publication evidence.
The reviewer baseline runner uses global baseline strength calibrated on
validation hosts; the separate standard/256-bit baselines use descriptive
per-host PSNR calibration. Only rows achieving the declared PSNR tolerance
should enter matched-condition claims.

## Executed validations

- `120 passed` (tests for Green-Quad, literature baselines, V2 audit and V3 runner).
- Ablation quick run completed; hyperparameter quick run completed.
- In a six-host smoke run, all **nine** planned smoke jobs were verified
  complete with `failed={}` and `incomplete=[]`, including restart/resume
  and matched-baseline comparison.
- Consolidated smoke ZIP passed `zipfile.ZipFile(...).testzip() is None`.
- **The full 11-job Kaggle run with a genuine 12-host external corpus was not
  executed in this environment.** It must be run on Kaggle before claiming
  completeness or publication-ready 12-host evidence.
