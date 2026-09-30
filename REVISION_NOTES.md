# Revision notes — post-embedding certificate fix

## P0

- Certificate is now evaluated on the **final rounded watermarked image**.
- Convex-hull convolution bound uses per-column perturbation-norm maxima before the nonlinear finite `r12` inequality.
- Fixed scalar rounding budget is replaced by a deterministic continuous-to-rounded `r12` group bound.
- Authenticated one-bit certificate mask records exactly which payload groups are certified.
- Added non-vacuous theorem regression over all configured extreme kernels and unseen convex mixtures.
- Added `configs/attacks/certified_convolution.yaml` for theorem-family experiments.

## P1

- Initial period selection minimizes payload-independent worst-bit continuous distortion rather than maximizing certificate margin.
- Uncertified groups escalate to larger feasible periods only after post-embedding certification; the entire image is re-convolved after each pass.
- Groups that still fail are explicitly marked uncertified instead of silently falling back to an ordinary certified status.
- Main comparison can tune baseline quantization strengths to the proposal's realized embedding PSNR (`match_psnr_to_proposed: true`).
- Embedding PSNR/SSIM and attacked-image PSNR/SSIM are separate CSV fields.
- README, mathematical note, method YAML, and attack terminology are synchronized with the implemented v3 path.

## P2

- One embedding is cached and reused for all attacks/seeds.
- Side-information overhead is reported, including period codes, certificate mask, HMAC and serialized metadata.
- `random_pixel_dropout` replaces the misleading `occlusion` name; contiguous `rectangular_occlusion` is also provided.
- `registered_rotation_resample` replaces the ambiguous `rotation_resample` name in experiment configs; the legacy alias remains for compatibility.
- Leakage script now checks both the period code alone and the joint period/certificate-status side code.

## Validation

- `python -m compileall -q src scripts tests`: pass.
- `pytest -q`: **22 passed**.
- Included `girl.bmp` / `watermark_1.png` real-image check: embedding PSNR is about **52 dB** with roughly **41%** certified groups under the default configuration (exact value varies with key).
- For an independent real-image run under the exact theorem family:
  - extreme `alpha=0.0`: certified BER = **0**;
  - unseen mixture `alpha=0.5`: certified BER = **0**;
  - extreme `alpha=1.0`: certified BER = **0**.

Overall BER may remain nonzero because uncertified groups are intentionally retained and reported separately. The theorem covers real-valued reflect-101 convolution in the configured convex hull; post-attack requantization and other attacks remain empirical.
