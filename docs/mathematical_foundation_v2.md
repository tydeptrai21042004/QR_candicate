# MC-CCQR spread-QIM: mathematical foundation (v4 hardware/path-certificate revision)

This note describes the mathematics implemented by `mc_ccqr_spread_qim_v4_hw_path_cert`. It separates the convolution theorem from empirical robustness outside the certified family and makes explicit the QR-free realization used by the hardware-oriented path.

## 1. Canonical QR carrier

For a 2×2 block `A=[a,b]`, with `a != 0`, the carrier is

\[
\phi(A)=r_{12}(A)=\frac{a^T b}{\|a\|_2}.
\]

If only `r12` changes by `delta` while the other canonical QR factors are fixed, the continuous reconstructed block changes by `delta q_1 e_2^T`. Hence

\[
\|\Delta A\|_F^2=\delta^2,
\qquad
\operatorname{MSE}_{2\times2}=\frac{\delta^2}{4}.
\]

This identity is exact before integer rounding.

The same identity gives a QR-free reconstruction rule. Since

\[
q_1=\frac{a}{\|a\|_2},
\]

an `r12` displacement `delta` can be applied directly as

\[
\boxed{b'=b+\delta\frac{a}{\|a\|_2}}.
\]

Thus canonical QR remains the mathematical reference, but no general QR engine is required in the streaming implementation.

## 2. Minimum-energy spread QIM

For `r` carriers `z=(r12_1,...,r12_r)` and unit vector `w`, define

\[
s=w^Tz.
\]

For bit `b`, let `q_b(s)` denote the nearest point of the corresponding binary QIM coset and `d=q_b(s)-s`. The encoder solves

\[
\min_\delta \frac12\|\delta\|_2^2,
\qquad
w^T\delta=d,
\qquad
\ell_i\le \delta_i\le u_i,
\]

where the interval constraints are the exact `r12` displacements that keep every reconstructed block in `[0,255]`.

The KKT solution has the clipped form

\[
\delta_i^*=\operatorname{clip}(\lambda w_i,\ell_i,u_i),
\]

with `lambda` chosen so that `w^T delta=d`. If no box constraint is active, `delta*=d w` and the continuous embedding energy is `d^2`.

## 3. Finite `r12` perturbation inequality

Let

\[
A=[a,b],\qquad A+E=[a+e,b+f].
\]

If `||e||_2 < ||a||_2`, the code uses

\[
|\phi(A+E)-\phi(A)|
\le
\|f\|_2+
\frac{\|b\|_2\,\|e\|_2}{\|a\|_2-\|e\|_2}.
\]

When the denominator condition fails, the bound is marked infinite and that group cannot be certified by this theorem.

## 4. Deterministic integer-rounding bound

The QIM target is reached in continuous arithmetic, but the exported watermark is uint8. The old implementation used a fixed scalar `rounding_feature_budget`; that is no longer used for certification.

For each group, let `A_i^c` be the continuous embedded block and `A_i^r` its final clipped/rounded block. The finite inequality above is applied to every pair `(A_i^c,A_i^r)`. With unit spread weights,

\[
\rho_{\rm round}
=\sum_i |w_i|\,\rho_i(A_i^c\to A_i^r).
\]

This is a deterministic group-specific theorem bound. The implementation also records the observed statistic shift only as a diagnostic.

## 5. Convolution certificate on the final watermarked image

Certification is performed **after all groups have been embedded and rounded**.

Let `Y` denote the final rounded watermarked channel and let the configured extreme kernels be

\[
h^{(1)},\ldots,h^{(L)}.
\]

The certified real-valued attack family is

\[
\mathcal H=\operatorname{conv}\{h^{(1)},\ldots,h^{(L)}\},
\]

with reflect-101 boundaries.

For one 2×2 watermarked block `A`, define the extreme perturbations

\[
E_\ell=(h^{(\ell)}*Y)_{\rm block}-A.
\]

For any convex mixture, linearity of convolution gives

\[
E=\sum_\ell \alpha_\ell E_\ell,
\qquad \alpha_\ell\ge0,
\qquad \sum_\ell\alpha_\ell=1.
\]

Therefore each perturbation-column norm satisfies

\[
\|e\|_2\le \max_\ell\|e_\ell\|_2,
\qquad
\|f\|_2\le \max_\ell\|f_\ell\|_2.
\]

Substituting these two maxima into the finite `r12` inequality produces a conservative block bound valid for **every convex mixture**, including the case where different extreme kernels maximize the two columns. The spread-group convolution bound is

\[
\rho_{\mathcal H}=\sum_i |w_i|\rho_i.
\]

This avoids the incorrect shortcut of merely taking the maximum of already-combined nonlinear extreme bounds.

### 5.1 Tighter two-extreme path theorem

When the certified family has exactly two extreme kernels, every attacked block lies on one scalar affine path. For block `i`, write

\[
B_i(\alpha)=(1-\alpha)B_i^{(0)}+\alpha B_i^{(1)},
\qquad 0\le\alpha\le1.
\]

Partition `[0,1]` into `J` equal intervals. For interval `j`, let `m_j` be the midpoint and `h=1/(2J)` the half-width. Define the midpoint block

\[
C_{i,j}=B_i(m_j).
\]

For every `alpha` in that interval,

\[
B_i(\alpha)-C_{i,j}=(\alpha-m_j)\bigl(B_i^{(1)}-B_i^{(0)}\bigr).
\]

Hence the first- and second-column perturbations obey

\[
\|e_{i,j}\|_2\le h\|a_i^{(1)}-a_i^{(0)}\|_2,
\qquad
\|f_{i,j}\|_2\le h\|b_i^{(1)}-b_i^{(0)}\|_2.
\]

Applying the finite `r12` inequality around `C_{i,j}` gives a rigorous local bound `L_{i,j}`. With spread weights `w_i`, define

\[
L_{g,j}=\sum_i |w_i|L_{i,j}.
\]

Let `S_Y` be the spread statistic of the final rounded watermarked blocks and `S(m_j)` the statistic at the interval midpoint. Then every `alpha` in interval `j` satisfies

\[
|S(\alpha)-S_Y|
\le
|S(m_j)-S_Y|+L_{g,j}.
\]

Therefore

\[
\boxed{
\rho_{\rm path}
=
\max_j\left(|S(m_j)-S_Y|+L_{g,j}\right)
}
\]

is a deterministic certificate for the complete two-kernel convex path. The implementation finally uses

\[
\boxed{
\rho_{\mathcal H}^{\rm v4}
=
\min\{\rho_{\rm generic},\rho_{\rm path}\}
}
\]

because both terms are independently valid bounds. Consequently the v4 final theorem can never be weaker than the previous generic certificate.

Crucially, this tighter theorem is evaluated **only after period allocation and watermarked pixels are frozen**. It is used to refine the authenticated certificate mask, not to reduce periods during the same run. Thus the v4 theorem does not change PSNR or the embedded image.

## 6. QIM survival condition

The continuous embedding statistic lies exactly on its intended binary QIM lattice point. The nearest decision boundary is `Delta/4` away. By triangle inequality, a sufficient survival condition is

\[
\rho_{\rm round}
+
\gamma\rho_{\mathcal H}
+
\rho_{\rm additive}
<
\frac{\Delta}{4},
\]

where `gamma >= 1` is the configured safety factor.

Only groups satisfying this inequality with finite component bounds are marked `certified=True`.

For the final v4 mask, `rho_H` is the tighter valid bound described in Section 5.1 when two extremes are configured; otherwise the generic hull theorem remains in force.

## 7. Period selection and explicit failure

The encoder no longer maximizes certificate margin, which previously biased selection toward the largest period.

It first chooses the **lowest worst-bit continuous embedding distortion** among range-feasible period candidates, without using the payload bit. After the complete image is rounded and certified, an uncertified group may escalate to the next larger feasible period. The complete watermarked image is re-convolved after each escalation pass because convolution couples neighboring blocks.

A global RGB SSE budget corresponding to `target_psnr_db` constrains escalation. If no available period can obtain a certificate within the configured passes/budget, the group remains embedded but is explicitly marked **uncertified** in the authenticated side information.

## 8. PSNR accounting

For an RGB image of size `H x W`, target `P0` dB corresponds to

\[
D(P_0)=3HW\,255^2\,10^{-P_0/10}.
\]

The embedder reports actual RGB SSE and embedding PSNR. The benchmark separately reports attack-distorted PSNR; the two are never conflated.

For fair robustness comparisons, the main comparison tunes baseline quantization strength to the proposal's realized embedding PSNR on the same host/watermark pair.

## 9. Side-information accounting

Authenticated proposal side information contains:

- period-table codes;
- one certificate-status bit per payload bit;
- metadata;
- a 256-bit HMAC-SHA256 tag.

The benchmark reports this overhead rather than comparing payload robustness while ignoring auxiliary information.

## 10. What is not certified

The theorem above covers real-valued reflect-101 convolution by kernels in the configured convex hull. The following remain empirical unless an additional theorem is supplied:

- JPEG/JPEG2000;
- post-attack integer requantization;
- random noise and impulse noise;
- median/nonlinear filtering;
- rescaling;
- registered rotate/inverse-rotate resampling;
- unknown rotation/translation/cropping;
- random pixel dropout or rectangular occlusion;
- generative/regeneration attacks.

The test suite therefore distinguishes **theorem regression** from general robustness experiments.

## 11. Hardware QIM decision

For statistic `s` and period `Delta`, let

\[
\phi=s\bmod\Delta.
\]

The two QIM cosets are separated by boundaries at `0`, `Delta/2`, and `Delta`. The hard decision is therefore exactly

\[
\hat b=
\begin{cases}
0,&0\le\phi<\Delta/2,\\
1,&\Delta/2\le\phi<\Delta.
\end{cases}
\]

The implementation uses the normalized distance to the nearest boundary as a triangular confidence. This has the same hard decision as the former sinusoidal score but needs only subtraction, comparison, `min`, and scaling in a hardware datapath.
