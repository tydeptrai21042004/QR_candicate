# CCQR v5 real-time edge execution path

## Goal

The mathematical method is unchanged: canonical 2x2 `r12`, spread-QIM, the same four periods, the same minimum-energy box projection, the same PSNR budget, and the same post-embedding convolution certificate. v5 only removes software work that does not belong in a per-frame edge datapath.

## Changes that are output-preserving

1. **Session-cached keyed permutation.** HMAC-SHA256 block ordering is computed once for a fixed `(resolution, key)` and reused. In hardware this is descriptor ROM/BRAM populated by the control processor.
2. **Vectorized selected-block decoder.** The decoder gathers all selected 2x2 blocks and evaluates `r12=(a^T b)/||a||` in a batch. QIM remains comparator/phase based.
3. **Batched spread-QIM candidate bank.** All four period candidates are prepared once from the original selected blocks. The common KKT solution `delta=d*w` is vectorized; only rare box-active groups use the bounded fallback.
4. **Batched certificate arithmetic.** Generic and two-extreme path bounds operate on `(group, repetition, 2, 2)` arrays instead of Python block objects.
5. **Cached final theorem inputs.** Final path tightening reuses the extreme-filter blocks from the last unchanged watermarked image; it does not convolve the frame again.
6. **Half-integer compatibility guard.** Rare exact `k+0.5` reconstruction points use the legacy QR rounding path so the optimized software output stays bit-exact with v4.

## Measured reference behavior

The uploaded v4 Kaggle benchmark reported about 44 s certified embedding and 0.54 s extraction for 512x512 images. On the development CPU used for the v5 refactor, the same full 64x64-payload Girl case is approximately 0.4--0.5 s for certified embedding and about 12--14 ms for warm-session extraction. These are software-reference measurements, not edge-board claims.

A full equivalence check against the original v4 implementation for `girl.bmp`, `watermark_1.png`, and key `edge-full-equivalence-2026` produced:

- identical watermarked image SHA-256;
- 0 differing pixels;
- 0 differing packed period-code bytes;
- 0 differing packed certificate-mask bytes.

## 512x512 at 30 fps hardware budget

For 512x512 frames, 30 fps is 7.864 Mpixel/s per frame traversal. The worst configured execution uses:

- one base-frame embedding traversal;
- at most four certificate traversals (`certificate_max_passes=4`);
- both 3x3 extreme kernels in parallel;
- no extra frame traversal for final path tightening.

Thus the conservative image-stream requirement is about `5 * 7.864 = 39.32 Mpixel/s`. A 100 MHz one-pixel-per-cycle FPGA/SoC pipeline has sufficient raw pixel-rate margin for this resolution. This is a cycle-budget feasibility argument; final real-time claims require synthesis and board measurements.

## Suggested edge partition

### Control plane (ARM/RISC-V CPU)

- HMAC/SHA-256 key handling and side-info authentication;
- generation of the keyed selected-block descriptor table once per session/key;
- configuration registers and output metadata.

### Streaming data plane (FPGA/ASIC/SIMD accelerator)

- 2x2 block assembly;
- `r12` dot/norm/reciprocal-square-root primitive;
- 5-lane spread accumulation;
- constant-period QIM and box checks;
- closed-form `b' = b + delta*a/||a||` update;
- two parallel 3x3 Gaussian filters;
- generic/path certificate arithmetic.

For 4096 payload groups with repetition 5, only 20,480 selected block statistics are needed. At 16 bits/statistic that is about 40 KiB of feature storage. Period codes plus the certificate mask are also small; the v4 benchmark serialized side information at roughly 18.8 kbit per image.

## What is and is not real-time today

- **Warm extraction/verification:** already real-time in the optimized Python reference at 512x512 on the development CPU.
- **Certified embedding in Python:** much faster but not yet 30 fps; do not claim CPU real-time from the Python benchmark.
- **Certified hardware embedder:** arithmetic and bandwidth are compatible with a pipelined 512x512/30-fps FPGA/SoC implementation, but publish a real-time hardware claim only after synthesis and on-board timing/resource/power measurements.
