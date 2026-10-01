# Zareian & Tohidypour (2013) adaptive QIM baseline

Reference: M. Zareian and H. R. Tohidypour, “Robust quantisation index modulation-based approach for image watermarking,” *IET Image Processing*, 7(5), 432–441, 2013. DOI: `10.1049/iet-ipr.2013.0048`.

This module implements the published image-domain procedure rather than a generic fixed-step QIM surrogate:

- non-overlapping **16×16** blocks ranked by entropy;
- two-level orthonormal **Haar DWT**;
- one bit per selected block in the 4×4 level-2 approximation vector;
- adaptive power-law step `Delta_i = Delta0 * mean(abs(X_i)) ** (1/gamma)`;
- RMS-vector QIM from Eqs. (2)–(4);
- gain estimate and minimum-distance detector from Eqs. (5)–(8);
- published side-information accounting: block-selection mask plus nominal 8-bit `Delta0`, `gamma`, and `xi` words.

The paper is a scalar-image method. For the color-host benchmark it is applied to the configured image channel. This is an explicit protocol adaptation, not an assertion that the paper originally embedded RGB images.

The native payload is at most one bit per 16×16 block: a 512×512 host therefore has **1024 carriers**. Use `configs/experiments/qim_control_comparison.yaml` (16×16 binary watermark, 256 bits) rather than the 64×64/4096-bit main QR comparison.

Scale note: the paper's reported `Delta0≈0.21` is used with normalized image
intensities. The implementation therefore performs its Haar/QIM equations on
`[0,1]` data and converts back to 8-bit pixels after inverse Haar. Applying
`0.21` directly to 0--255 coefficients would make the mark disappear under
integer pixel rounding and is not a faithful reproduction of that operating
point.
