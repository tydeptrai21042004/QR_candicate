# v4 validation log

The v4 revision was accepted only when it preserved existing embedding behavior or made a mathematically monotone improvement.

## Automated regression

- `python -m compileall -q src scripts tests`: pass.
- `pytest -q`: 44 passed.
- The theorem regression is non-vacuous and checks configured extreme kernels plus unseen convex mixtures.

## QR-free reconstruction equivalence

A dedicated regression compares the algebraic update

`b' = b + delta * a / ||a||`

against canonical `Q @ R_new` on 5,000 random 2x2 blocks. Maximum floating discrepancy is below `1e-10` and rounded output mismatches are zero.

For the full 512x512 `girl.bmp` / `watermark_1.png` case with key `v4-full-girl`, disabling only the new final path-tightening theorem produces exactly the same watermarked uint8 image as the previous corrected v3 repository:

- changed pixels: 0;
- maximum pixel difference: 0;
- SHA-256: `c4b1f12f27bab2c06e1640a9d38dca38f9fe800a95c16de5329ac993a0110892`.

## Full Girl certificate check

With the final v4 path theorem enabled on the same full case:

- embedding PSNR: approximately 52.068 dB;
- clean BER: 0;
- generic v3 certified fraction: 0.42334;
- v4 final certified fraction: 0.79102;
- newly certified groups: 1,506 / 4,096.

The final theorem is run after period allocation is frozen, so this certificate increase does not alter the embedded image.

## Six-host small-payload screening

An 8x8 diagnostic payload was used only as a fast host-diversity smoke test for the theorem. Generic -> final certified fractions were:

| Host | Generic | v4 path-tightened |
|---|---:|---:|
| airplane | 0.000 | 0.562 |
| girl | 0.516 | 0.781 |
| lenna | 0.000 | 0.688 |
| manhattan | 0.016 | 0.578 |
| pepper | 0.000 | 0.500 |
| safari | 0.031 | 0.641 |

These values are diagnostic, not a replacement for the full paper benchmark.

## Ideas explicitly rejected

A local five-block carrier layout improved several blur/resampling attacks but caused clean embedding failures and worsened JPEG on at least one host, so it was not merged.

A six-block local layout lowered PSNR and worsened JPEG/salt-and-pepper in testing, so it was not merged.

A 3x3 impulse-gated median decoder improved mean salt-and-pepper BER in a six-host 8x8 screening, but regressed Airplane and Manhattan. Because the requirement is not to trade one host for another, it was removed from the deliverable.

The final v4 patch therefore contains no decoder denoiser, carrier-layout change, larger period, DWT/DCT/SVD, or learned component.
