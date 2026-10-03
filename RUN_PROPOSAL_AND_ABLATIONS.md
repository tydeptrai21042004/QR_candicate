# Run the proposal and all major ablations

## 1. Install

```bash
pip install -r requirements.txt
pip install -e .
```

## 2. Run the full proposal on one host/watermark

```bash
python scripts/run_method.py roundtrip \
  --method proposed \
  --config configs/methods/proposed.yaml \
  --host data/hosts/classical/girl.bmp \
  --watermark data/watermarks/watermark_1.png \
  --key "research-key" \
  --out runs/proposal/girl_wm1.png
```

## 3. Run all major ablations

```bash
python scripts/run_ablation.py \
  --config configs/experiments/ablation.yaml \
  --key "research-key" \
  --run-dir runs/ablation
```

Outputs:

- `runs/ablation/ablation.csv`: every individual evaluation.
- `runs/ablation/ablation_summary.csv`: per-variant/per-attack mean, std, count.
- `runs/ablation/ablation_overall.csv`: overall variant means.

Run only selected variants by repeating `--variant`:

```bash
python scripts/run_ablation.py \
  --config configs/experiments/ablation.yaml \
  --key "research-key" \
  --run-dir runs/ablation_selected \
  --variant full_ccqr \
  --variant no_final_path_tightening
```

## 4. Major ablations included

- `full_ccqr`: complete proposal.
- `fixed_period_32`: removes adaptive multi-candidate QIM period selection/escalation.
- `no_spread_repetition`: sets repetition to 1, removing multi-block spread diversity.
- `repetition_3`: intermediate repetition sensitivity point.
- `no_convolution_bank`: removes the configured convolution uncertainty family.
- `no_certificate_escalation`: keeps certification but limits it to one pass.
- `no_final_path_tightening`: removes the v4 two-extreme path bound while preserving pixels and period allocation.

## 5. Hyperparameters

The method has genuine algorithmic hyperparameters. The most important ones are:

- `repetition`: number of 2x2 carrier blocks per payload bit. Default `5`.
- `period_candidates`: candidate QIM periods. Default `[24, 28, 32, 36]`.
- `max_group_mse`: local embedding distortion feasibility limit. Default `220`.
- `target_psnr_db`: global RGB embedding distortion budget used during adaptive escalation. Default `50 dB`.
- `convolution_gaussian_sigmas`: extreme kernels defining the certified convolution family. Default `[0.50, 0.75]`.
- `convolution_safety_factor`: multiplier on the theorem convolution bound. Default `1.0`.
- `additive_feature_budget`: additional feature-domain uncertainty reserve. Default `1.0`.
- `certificate_max_passes`: maximum adaptive re-certification/escalation passes. Default `4`.
- `certificate_path_subdivisions`: resolution of the rigorous two-extreme piecewise path theorem. Default `8`.

Protocol/fixed settings that should normally NOT be presented as tuned hyperparameters:

- `block_size=2`: mathematically fixed by the current R12 derivation.
- `watermark_size=64`: payload/evaluation setting.
- `channel=0`: embedding-channel protocol choice.
- `arnold_iterations=10`: scrambling protocol setting.
- `certificate_final_tighten`: a method-component switch for ablation, not a scalar parameter to optimize.
- `nlm.*`: disabled in the default proposal; do not tune it unless NLM is explicitly made part of the claimed method.

## 6. Recommended paper reporting

Use the default configuration as the primary proposed method. Report component ablations separately from scalar sensitivity. For sensitivity, `repetition` and the QIM period set are the two most important parameters; certificate-only parameters should be evaluated primarily with `certified_fraction` and runtime, while embedding parameters should additionally report PSNR/SSIM and BER/NC under attacks.
