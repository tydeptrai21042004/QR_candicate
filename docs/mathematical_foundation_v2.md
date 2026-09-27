# MC-CCQR Spread-QIM: mathematical foundation (v2)

This note records the mathematics implemented by the proposal path. It deliberately separates **proved/certified statements** from empirical attack robustness.

## 1. Exact QR carrier
For a 2x2 block `A=[a,b]`, the proposal uses

\[
\phi(A)=r_{12}(A)=\frac{a^Tb}{\|a\|_2}.
\]

Changing only `r12` by `delta` while keeping the canonical QR factors otherwise unchanged produces exact block squared error `delta^2` before integer rounding.

## 2. Minimum-energy spread QIM
For a group of `r` QR carriers `z=(r12_1,...,r12_r)` and a unit vector `w`, the decoder statistic is `s=w^T z`. For the selected bit lattice point `q_b(s)`, set `d=q_b(s)-s` and solve

\[
\min_\delta \frac12\|\delta\|_2^2,
\qquad w^T\delta=d,
\qquad \ell_i\le\delta_i\le u_i.
\]

The KKT solution is

\[
\delta_i^*=\operatorname{clip}(\lambda w_i,\ell_i,u_i),
\]

with the scalar `lambda` chosen so that `w^T delta=d`. If no box constraint is active, `delta^*=d w` and the exact continuous embedding energy is `d^2`. This replaces independent repetition, whose energy grows with the number of repeated copies.

## 3. Finite nonlinear R12 perturbation bound
For `A=[a,b]`, `A+E=[a+e,b+f]`, if `||e||_2 < ||a||_2`, the implementation uses

\[
|\phi(A+E)-\phi(A)|
\le
\|f\|_2+
\frac{\|b\|_2\,\|e\|_2}{\|a\|_2-\|e\|_2}.
\]

This is finite/non-asymptotic; it is not a first-order approximation.

## 4. Unknown convolution family
The certified attack set is

\[
\mathcal H=\operatorname{conv}\{h^{(1)},\ldots,h^{(L)}\},
\]

implemented with reflect-101 boundary convolution. Image/block perturbations depend affinely on `h`. The finite bound above is convex in the column perturbation norms, so the weighted group bound is convex in `h`. Therefore its maximum over the convex hull is attained at an extreme kernel. Evaluating only the configured extreme kernels certifies **every convex mixture** in the hull.

For the spread statistic,

\[
|\widetilde s-s|\le \sum_j |w_j|\rho_j(h).
\]

A sufficient QIM survival condition is

\[
\rho_{\mathcal H}+\rho_{\rm additive}+\rho_{\rm round}<\Delta/4.
\]

## 5. Unknown-filter minimax allocation
`robust_allocation.py` solves

\[
\max_{p\in\Delta_J}\min_{\ell=1,\ldots,L}(S p)_\ell,
\]

with optional per-carrier caps and a perceptual cost budget. This is a linear program and is the reusable robust counterpart for allocating watermark strength across a fixed carrier/frequency dictionary when the exact convolution filter is unknown.

The present spatial equal-spread implementation uses uniform unit weights so extraction needs no additional host-dependent weight side information. The LP is provided for the next fixed-frequency/DCT carrier extension.

## 6. 50 dB accounting
For an RGB image of size `H x W`, a target `P0` dB corresponds to squared-error budget

\[
D(P_0)=3HW\,255^2\,10^{-P_0/10}.
\]

The embedder reports both this budget and actual RGB SSE/PSNR. With the default spread-QIM periods capped at 36, the design is tuned toward the requested 50 dB regime; this is an empirical operating target unless the reported `actual_rgb_sse <= target_sse_budget` condition is satisfied for a particular image.

## 7. What is *not* certified
JPEG/WebP, cropping, rotation, rescaling, diffusion/regeneration, asymmetric non-convolution attacks, and attacks outside the configured convolution hull remain empirical stress tests. They must not be described as covered by the convolution theorem.
