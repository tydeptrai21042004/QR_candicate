# Zareian & Tohidypour (2013) adaptive QIM baseline

Reference: M. Zareian and H. R. Tohidypour, “Robust quantisation index modulation-based approach for image watermarking,” *IET Image Processing*, 7(5), 432-441, 2013. DOI: `10.1049/iet-ipr.2013.0048`.

This is intentionally a **semi-blind / side-information-assisted** baseline. The implementation keeps the decoder information required by the paper instead of converting the method into a blind one:

- non-overlapping **16x16** blocks ranked by entropy;
- two-level orthonormal **Haar DWT**;
- one bit per selected block in the 4x4 level-2 approximation vector;
- adaptive power-law step `Delta_i = Delta0 * mean(abs(X_i)) ** (1/gamma)`;
- RMS-vector QIM from Eqs. (2)-(4);
- gain estimate and minimum-distance detector from Eqs. (5)-(8);
- transmitted selected-block map plus `Delta0`, `gamma`, and `xi`;
- the paper's unknown-rotation synchronization: test candidate angles from **-10 to +10 degrees in 0.5-degree steps**, inverse-rotate the received image, recompute its highest-entropy blocks, and select the angle with greatest overlap with the transmitted original block map before decoding.

The original method is scalar-image based. For this repository's color-host benchmark, it is applied to the configured image channel. That is a declared pipeline adaptation; the side-information model, carrier, equations, and native capacity are retained.

The native payload is at most one bit per 16x16 block: a 512x512 host therefore has **1024 carriers**. Use `configs/experiments/qim_control_comparison.yaml` (16x16 binary watermark, 256 bits) rather than forcing it into the 64x64/4096-bit main comparison.

Scale note: the paper's reported `Delta0≈0.21` is interpreted on normalized image intensities. Haar/QIM therefore runs on `[0,1]` data and converts back to 8-bit pixels after inverse Haar.
