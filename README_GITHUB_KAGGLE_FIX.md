# GitHub / Kaggle repair for QR_candicate (2026-10-10)

This repository snapshot lacked `src/qrwatermark/proposed/fused_convqr_v6.py` and `scripts/benchmark_fcqr_v6.py`. The missing module prevented pytest collection and caused v7-v9 imports to fail. The legacy carrier benchmark was also not runnable.

## Applied changes

* `src/qrwatermark/proposed/block_carrier.py`: a deterministic keyed 4x4 block selector and block gatherer that replaces ONLY the missing shared helper functionality for v7-v9.
* `src/qrwatermark/proposed/integer_convolution_v7.py`, `integer_convolution_pair_v8.py`, `integer_convolution_quad_v9.py`: import the selector from the independent helper.
* `src/qrwatermark/proposed/method.py`: FCQR-v6 is lazily imported and unavailable cases produce a descriptive error without preventing other methods from loading.
* `scripts/benchmark_integer_carriers.py`: executable replacement independent of the missing historical v6 benchmark; defaults to v5, v7, v8, v9. Option `--include-v6` is allowed only after supplying the genuine v6 source and configuration.
* `tests/test_recovered_block_carrier.py`: deterministic permutation, disjointness, capacity, and gathering tests.
* `kaggle/QR_KAGGLE_GITHUB_ONE_CELL.py`: fresh GitHub clone with **no hidden automatic source patching**, fail-fast validation, PSNR-matched paper baselines, 26 attack conditions, proposal family, ablations, runtime, checkpointed outputs. The `.ipynb` contains the same code in a single cell.

**Scientific provenance warning:** the historical `fused_convqr_v6.py` was not supplied. The repaired block selector is a **new reconstruction**, not the original v6 mapping; it must not be described as mathematically or numerically identical to that source. FCQR-v6 is not implemented by this patch, and prior results for v7-v9 must be rerun under the new mapping. Tests demonstrating clean recovery are not a substitute for full robustness benchmarking. The script's timing peak counts only Python `tracemalloc` allocations, not native-process peak RSS.

## GitHub upload

From the repository root, overlay the contents of `QR_GITHUB_CHANGED_FILES.zip` preserving its paths and commit them. Or replace your repository with the contents of `QR_GITHUB_FULL_REPOSITORY.zip` (contains the full supplied snapshot plus these changes; no `.git` metadata). In particular, preserve the new `src/qrwatermark/proposed/block_carrier.py` file.

```bash
python -m pip install -e ".[test]"
python -m pytest -q tests
python scripts/benchmark_integer_carriers.py --out /tmp/qr-smoke --smoke --repeats 1
```

## Kaggle

1. Commit the patch to GitHub `tydeptrai21042004/QR_candicate` main branch.
2. Set Kaggle Internet ON. The pipeline uses CPU; GPU is not required.
3. Copy `kaggle/QR_KAGGLE_GITHUB_ONE_CELL.py` to a single Kaggle Python cell, or import the matching notebook.
4. The runner clones the repository and verifies the required files. If GitHub is still missing the patch, it stops with a clear error before trying any experiments.
5. Final/periodic archive: `/kaggle/working/QR_ALL_COMPLETED_RESULTS.zip`, with `run_state.json`, traceback logs and collected CSV results. It checkpoints incrementally and stops new jobs before session deadline. An incomplete job is not counted as a successful experiment.

`QR_SMOKE_ONLY=1` in environment (or change `SMOKE_ONLY = True` near the top of the cell) runs a small preflight. Normal full mode uses 6 supplied BMP hosts, 2 supplied watermark PNGs, 5 paper baselines, v9-v5 comparisons excluding unprovided FCQR-v6, 25 attacked settings + clean, ablations and runtime. Long Kaggle execution has not been claimed completed locally.
