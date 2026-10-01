# Hardware deployment note — v4 HW-path certificate

This note describes a hardware realization of the implemented proposal without changing the watermarking law, payload, QIM periods, or PSNR target.

## 1. QR-free 2x2 carrier datapath

For a 2x2 block `A=[a,b]`, canonical QR gives

\[
r_{12}=\frac{a^Tb}{\|a\|_2},\qquad q_1=\frac{a}{\|a\|_2}.
\]

If the spread-QIM solver requests an `r12` displacement `delta`, the exact reconstructed block is

\[
b'=b+\delta q_1=b+\delta\frac{a}{\|a\|_2}.
\]

Therefore the production datapath does not require a matrix QR engine. A small pipeline needs:

- four pixel registers for one 2x2 block;
- two multipliers plus an adder for `a^T b`;
- two squares plus an adder for `a^T a`;
- reciprocal square root (LUT/Newton/CORDIC implementation choice);
- two multiply-adds for reconstruction.

The software implementation keeps a rare half-integer rounding compatibility fallback so existing NumPy-generated uint8 watermarks remain bit-exact. That fallback is a software reproducibility detail, not a requirement of the mathematical datapath.

## 2. QIM decoder without trigonometry

For period `Delta`, let

\[
\phi=s\bmod\Delta.
\]

The binary decision is

\[
\hat b=0\quad\text{for }0\le\phi<\Delta/2,
\qquad
\hat b=1\quad\text{for }\Delta/2\le\phi<\Delta.
\]

A triangular confidence is the normalized distance to the nearest decision boundary. The implementation therefore needs only remainder/subtraction, comparisons, `min`, and fixed scaling; it does not require `sin()`.

## 3. Minimum-energy projection

The unconstrained solution is

\[
\delta=d w,
\]

where `w` is unit norm. The existing implementation first checks this closed-form candidate against all pixel-range intervals. Only a failed interval check enters the bounded projection fallback. This maps naturally to a hardware fast path plus a small controller for boundary cases.

## 4. Convolution certificate engine

Only the configured 3x3 extreme kernels are convolved with the final rounded watermarked channel. With two extremes, every certified attack lies on the scalar path

\[
H_\alpha=(1-\alpha)H_0+\alpha H_1,\qquad 0\le\alpha\le1.
\]

The v4 final theorem partitions this one-dimensional path into a small fixed number of intervals. It uses no extra image convolution beyond the two extreme responses: interval-center attacked blocks are affine combinations of those two responses.

The final tight theorem is intentionally run only after all embedding periods are frozen. It can increase the authenticated certified mask, but cannot modify watermarked pixels, PSNR, payload, or period allocation.

## 5. Suggested hardware/software split

**Streaming dataplane (FPGA/ASIC):**

- 2x2 block assembly;
- closed-form `r12` extraction;
- spread accumulation;
- QIM target/decision;
- minimum-energy fast path and bounded fallback;
- block reconstruction;
- 3x3 extreme convolution;
- per-group certificate arithmetic.

**Control plane (CPU/MCU/SoC processor):**

- HMAC-SHA256;
- keyed permutation/descriptor generation;
- side-information serialization;
- experiment/logging functions.

This split avoids putting a sorting-based HMAC permutation generator or general scientific-Python routines in the streaming datapath while preserving the exact security semantics of the repository.

## 6. Fixed-point starting point

For 8-bit pixels,

\[
|r_{12}|\le \|b\|_2\le255\sqrt2<361.
\]

A signed range of at least `[-512,512)` is therefore sufficient for the carrier itself. A practical RTL study should sweep fractional precision and reciprocal-square-root precision and compare hard QIM decisions against the floating reference. Do not claim a final word length until synthesis/RTL regression is run on all hosts and attacks.
