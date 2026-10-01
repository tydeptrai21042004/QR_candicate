# Revision notes — v4 hardware/path certificate

## Design rule

The v4 revision was constrained to preserve the existing watermarking identity: 2x2 canonical `r12`, equal-unit-norm spread QIM, the same period table, the same PSNR budget, the same payload mapping, and the same authenticated side-information semantics. Candidate ideas that improved one attack but hurt clean behavior, PSNR, or another host were rejected.

## Safe hardware changes

- Replaced the normal `Q @ R_new` reconstruction path with the exact closed form `b' = b + delta*a/||a||`.
- Kept canonical QR only as a degenerate/half-integer software compatibility fallback.
- Added a regression proving rounded reconstruction equivalence on random blocks.
- Replaced the sinusoidal spread-QIM hard decision/confidence with a comparator plus triangular margin; hard decisions are regression-tested against the former rule.
- Added `docs/hardware_deployment_v4.md` and `scripts/run_hardware_sanity.py`.

## New mathematical robustness contribution

- Added a deterministic piecewise theorem specialized to the exact two-kernel convex path.
- The path theorem works from the same two extreme convolution responses; it does not require extra image convolutions.
- The final bound is `min(generic_hull_bound, two_extreme_path_bound)`, so it cannot weaken the v3 theorem.
- The v3 generic certificate is still used during period escalation.
- The v4 path theorem is applied only after the final pixels/periods are frozen. It can only increase the authenticated certificate mask and cannot alter PSNR, payload, periods, or the watermarked image.

## Rejected changes

- Local five-block grouping: rejected because it caused clean embedding failures and a JPEG regression despite improving several blur/resampling attacks.
- Local six-block grouping: rejected because it reduced PSNR and worsened JPEG/salt-and-pepper.
- Global phase compensation: rejected because gains were small and some cases regressed.
- 3x3 impulse-gated median decoder: rejected after six-host screening because Airplane and Manhattan regressed under salt-and-pepper even though the mean improved.
- Larger QIM periods, DWT/DCT/SVD, CNNs, and heavy optimization were not introduced.

## Validation

- `python -m compileall -q src scripts tests`: pass.
- `pytest -q`: **44 passed**.
- Full 512x512 `girl.bmp` / `watermark_1.png` check:
  - embedding PSNR: about **52.068 dB**;
  - clean BER: **0**;
  - generic certified fraction: **0.42334**;
  - final v4 certified fraction: **0.79102**;
  - newly certified groups: **1,506 / 4,096**.
- With v4 final tightening disabled, the QR-free implementation produces the exact same full Girl uint8 watermarked image as the previous corrected v3 repository for the validation key: **0 changed pixels**, maximum difference **0**, SHA-256 `c4b1f12f27bab2c06e1640a9d38dca38f9fe800a95c16de5329ac993a0110892`.

See `docs/v4_validation.md` for the host-diversity smoke test and rejected-idea record.
