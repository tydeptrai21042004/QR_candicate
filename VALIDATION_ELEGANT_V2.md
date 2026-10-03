# ME-CQR-QIM v2 validation snapshot

These are smoke/regression measurements, not a substitute for the full paper attack campaign. They are included so the code changes can be checked immediately.

## Environment-level correctness

```text
69 passed
```

The tests include clean round-trip, authenticated selector packing, no-repetition invariants, certificate/pixel independence, legacy regression tests, and the existing baseline/QR tests.

## Six-host integrated comparison

Data: all six images under `data/hosts/classical`, both supplied 64x64 watermarks. Quick attacks:

- clean;
- Gaussian blur, sigma = 1.0;
- Gaussian noise, variance = 0.003, seed = 0;
- JPEG, quality = 50.

Averages across 6 hosts x 2 watermarks:

| Metric | ME-CQR-QIM v2 | legacy spread-QIM |
|---|---:|---:|
| embedding PSNR | 52.381 dB | 50.822 dB |
| embedding SSIM | 0.997425 | 0.996227 |
| clean BER | 0.000000 | 0.000000 |
| blur BER | 0.070496 | 0.364665 |
| noise BER | 0.239726 | 0.457214 |
| JPEG BER | 0.149251 | 0.468424 |
| certified fraction | 0.960958 | 0.602478 |
| logical side information | 8,448 bits | 12,544 bits |
| measured embed time in this sweep | 0.0581 s | 0.2989 s |

On this quick sweep, v2 reduced blur BER by about 80.7%, noise BER by 47.6%, JPEG BER by 68.1%, logical side information by 32.7%, and measured embedding time by about 80.6% relative to the legacy repeated implementation. Timing varies by machine and first-call cache/warm-up effects, so use the runtime experiment for paper reporting.

## Warmed single-image timing sanity check

Girl + watermark_1, eight embeddings, median of runs 3-8:

| Mode | Warm median embed time |
|---|---:|
| ME-CQR-QIM v2 + certificate audit | 0.0332 s |
| ME-CQR-QIM v2 fast profile | 0.0237 s |
| legacy repeated proposal | 0.1942 s |

The fast profile and certified profile produce the same watermarked pixels. Certification is a post-embedding audit only.

## Expanded ablations

The included one-host smoke matrix contains 10 variants. Important observations from that run:

- `full_mecqr`: PSNR 52.257 dB, clean BER 0, blur BER 0.0387, JPEG BER 0.1105.
- `pool_1_no_choice`: clean BER stays 0 but PSNR falls to 44.424 dB and robustness degrades, showing that candidate choice is not redundant.
- `pool_8`: PSNR 57.454 dB, clean BER 0, blur BER 0.0062, but selector overhead rises to 12,544 logical bits; this is why pool-4 remains the conservative default.
- `period_48` vs `period_80` exposes the expected imperceptibility/robustness trade-off using one global parameter rather than per-bit rules.
- `no_certificate` has identical PSNR/BER to `full_mecqr`; only certificate reporting and runtime change.
- `first_feasible_selector` performs much worse than minimum-energy selection, supporting the proposed argmin objective.

Exact CSV summaries are under `validation/elegant_v2/`.
