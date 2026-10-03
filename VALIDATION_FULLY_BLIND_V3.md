# Fully Blind v3 validation

This validation is for the default `blind_mecqr_qim_v3` implementation.

## Clean all-host check

Dataset included in the repository:

- 6 classical hosts
- 2 binary 64x64 watermarks
- key: `research-key`
- `qim_period = 48`
- `blind_margin_ratio = 1/8`

Across all 12 host/watermark pairs:

| Metric | Mean |
|---|---:|
| PSNR | 51.3381 dB |
| SSIM | 0.996402 |
| BER | 0.000000 |
| NC | 1.000000 |
| logical side information | 0 bits |
| serialized side information | 0 bits |
| certified fraction | 0.5451 |

Every embedding returned `side_info=None`.

The first embedding call includes permutation/cache warm-up. Subsequent per-image embedding calls in this run were generally about 0.02--0.05 s on the validation environment; extraction was about 0.009--0.013 s. These are software timings, not hardware throughput claims.

Raw data: `validation/blind_v3_clean_allhosts.csv`.

## Key-only CLI check

The watermarked Girl image was extracted successfully using only:

- the watermarked image,
- `research-key`, and
- `configs/methods/proposed.yaml`.

No `--side-info` and no reference `--watermark` argument were supplied to the extraction command.

## Blind ablation smoke check

A smaller Girl + two-watermark smoke run used clean, Gaussian blur sigma 1, Gaussian noise variance 0.003, and JPEG quality 50. Every blind-v3 ablation reported exactly **0 side-information bits**.

Clean BER was 0 for all tested blind-v3 variants. The safe-set projection had substantially better imperceptibility than fixed-carrier center-QIM in this smoke test (about 50.36 dB vs 47.20 dB PSNR for the selected key/settings).

The robustness trade-off must be reported honestly: removing selector side information makes the method materially weaker under several strong attacks, especially JPEG 50. Do not compare v3 and selector-based v2 without also reporting the side-information difference.

Raw smoke data:

- `validation/blind_v3_ablation_smoke.csv`
- `validation/blind_v3_ablation_smoke_overall.csv`

## Regression suite

```text
74 passed
```

The v3-specific tests verify zero side information, image+key-only extraction, key-determined carrier locality, safe-set energy dominance over center-QIM, report-only certification, and wrong-key failure.
