# Change-only patch — one proposed method

## Apply

1. Unzip this archive **inside your existing repository root**. Accept replacements.
2. Run `python APPLY_SINGLE_PROPOSAL.py` to delete old proposal modules, deprecated proposal tests, and obsolete experiment scripts. `python APPLY_SINGLE_PROPOSAL.py --dry-run` previews deletions.
3. Install/test: `python -m pip install -e ".[test]" && python -m pytest -q`.
4. Run final full ablation: `python scripts/run_green_quad_ablation.py --mode full --repeats 250 --include-greedy-control --output results/green_quad_final`.

This is **not** a full repository ZIP; it contains changed/added files, the deletion manifest, and experiment evidence. It preserves all third-party literature baseline source implementations and shared attacks/metrics.

## Final algorithm and tests

- Only production proposal: `green_quad_min_energy`, **BGR green index 1**, period **66**, Arnold **0**, fixed-centre **exact integer L2**.
- Fully blind, no host in extraction, no ECC, no repeated bits, no side info.
- Cached deterministic keyed block mapping reduces per-frame overhead without changing recovered bits.
- Final tests: 101 passing locally (the source code must be rechecked on your machine).
- Six unique images × two binary watermarks, 25 attacked conditions each = 300 attacked cases, plus 12 clean; five-key clean gate = 60 cases.
- Full results: attacked mean BER **0.2398885**, mean NC **0.7810372**, min PSNR **50.4481 dB**, mean SSIM **0.9967521**, clean BER **0**.
- FPS depends on CPU. The included `runtime.csv` measures the local test platform; **do not report it as Kaggle FPS without Kaggle rerun**.
- Ablation: period64 BER 0.24436; period68 BER 0.23620 but min PSNR only 50.11; Arnold10 BER 0.24888; blue BER 0.26787; red BER 0.25987; nonoptimal greedy PSNR 44.53 dB and ~112 FPS (not production-ready).

## Limits

No proof of NC >=0.90, SSIM >=0.998, geometric invariance, or major novelty beyond prior integer QIM approaches. The benchmark has six, not 12, distinct hosts. Parameter selection used this family of tests historically; new independent holdout is needed to claim generalization.
