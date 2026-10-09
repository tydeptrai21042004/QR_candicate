# Experimental proposal: Uniformly Conditioned Convolution–QR QIM (ConvQR-v5)

**Status:** research prototype only. This is a different watermarking law and a different
keyed carrier mapping from v4. It has no established superiority in attacked-image BER,
or proof that its mathematical combination is novel relative to all published work.
The default `proposed.yaml` and five paper baselines are unchanged.

## Mathematical motivation

For each keyed 2-by-2 block `A=[a,b]`, let `a=(a0,a1)^T`, `b=(b0,b1)^T`.
Take `q=a/||a||_2`; if `a=0`, define `q=(1,0)^T`. Since pixel intensities are
nonnegative, `q0,q1 >= 0`. Define two output channels:

\[
  u=q^\top b=r_{12},\qquad
  v=(b_0-b_1)/\sqrt 2=(h*b)_{\mathrm{strided}},
  \qquad h=(1,-1)/\sqrt 2.
\]

This makes `v` an actual two-tap Haar high-pass convolution; `u` is the
canonical first QR coefficient without forming `Q` or `R` as matrices.
Both observations encode the **same bit**, one 2-by-2 block per bit.
The first column `a` is not modified.

For bit `m in {0,1}`, the corresponding QIM codebook is
`{ (1/4 + m/2 + k) Delta : k in Z }` for each channel, with independent
periods `Delta_u,Delta_v`.

With target shifts `d_u=u_target-u`, `d_v=v_target-v`, we solve

\[
\begin{bmatrix}q_0&q_1\\1/\sqrt2&-1/\sqrt2\end{bmatrix}
\begin{bmatrix}\delta b_0\\\delta b_1\end{bmatrix}
=\begin{bmatrix}d_u\\d_v\end{bmatrix}.
\]

The exact continuous inverse is

\[
  \delta b_0=\frac{d_u+\sqrt 2 q_1d_v}{q_0+q_1},\qquad
  \delta b_1=\frac{d_u-\sqrt 2 q_0d_v}{q_0+q_1}.
\]

### Proposition: uniform conditioning (proved)

Write the displayed coefficient matrix as `W(q)`. Its Gram matrix is

\[
  W W^\top=\begin{bmatrix}1&t\\t&1\end{bmatrix},
  \qquad t=(q_0-q_1)/\sqrt 2.
\]

Since `q0^2+q1^2=1` and `q0,q1 >= 0`, `|t| <= 1/sqrt(2)`.
Therefore its eigenvalues are `1+|t|,1-|t|`, and

\[
   \kappa_2(W) =\sqrt{\frac{1+|t|}{1-|t|}}
   \le 1+\sqrt 2.
\]

This bound is **sharp** at `(q0,q1)=(1,0)` and `(0,1)`.
Consequently the two-feature embedding never suffers ill-conditioned
convolution–QR coupling on nonnegative image blocks.

The continuous embedding is the unique solution of the two linear feature
constraints. Thus it minimizes Euclidean perturbation among all such solutions.
In the implemented uint8 algorithm, integer rounding and clipping mean this
continuous constrained minimum is **not** automatically the optimal integer
solution. A bounded integer closure repairs exceptional cases; remaining
failures are measurable by the reported per-feature clean match fractions.

### Corollary: continuous-domain distortion bound

Because each target is the nearest center of the required QIM coset,
`|d_u| <= Delta_u/2` and `|d_v| <= Delta_v/2`. By the smallest
singular value of the coupling operator,

\[
  \|\delta b\|_2^2
  \le\frac{\Delta_u^2+\Delta_v^2}{4(1-1/\sqrt 2)}.
\]

Summing over `B` disjoint selected blocks gives a deterministic upper bound
on total **continuous, unconstrained** embedding SSE. It does not hold
unchanged after clipping, integer rounding, or a lattice-repair change;
those must be separately bounded/measured.

### Lemma: additive-noise carrier stability

Let perturbations of the two image columns be `e_a,e_b`, where
`||e_a||_2 < ||a||_2` and `a` is nonzero. For the Haar convolution carrier,

\[
  |v(b+e_b)-v(b)| \le \|e_b\|_2.
\]

For the QR carrier, writing `q'=normalize(a+e_a)`, the conservative bound

\[
  |q'^\top(b+e_b)-q^\top b|
  \le \|e_b\|_2 +
  \frac{2\|b\|_2\|e_a\|_2}{\|a\|_2-\|e_a\|_2}
\]

follows from Cauchy-Schwarz and the normalized-vector perturbation
inequality. If these bounds are below the relevant QIM decision margins,
the respective bit decisions are certified under that stated additive
perturbation. This is **not** a Gaussian-blur/JPEG certificate: those
attacks can produce much larger, image-dependent feature shifts.

### Complexity

Let `B` be the payload bit count and `N=floor(H/2)floor(W/2)` the total
candidate blocks. The pre-existing v4 carrier-selection path builds HMAC
records for all `N` blocks and sorts them (`O(N log N)` time, `O(N)`
permutation storage). Experimental v5 uses **HMAC-keyed Floyd sampling**
plus a separate keyed Fisher-Yates shuffle to select `B` distinct blocks:
`O(B)` expected time and `O(B)` selection storage. The algorithm has
`O(B)` embedding and extraction arithmetic, with per-block QR and Haar
features computed from four 8-bit pixels. It avoids trigonometric,
hyperbolic, matrix-QR, and full-image convolution operations.

The image and returned image copy still require **O(HW)** memory. The
`O(B)` claim is for selection/working arrays and does not mean total peak
memory is independent of image size. The fixed two-tap kernel is evaluated
at selected locations only; using full-frame FFT would add overhead.

### Why this is not yet a "breakthrough" claim

Uniform conditioning is a precise proved property of the proposed coupling,
but a claim of research novelty requires stronger prior-art review. Fast
HMAC sampling is a standard algorithm, not new mathematics. Performance
improvements depend on real measurements, image set, payload, Python/CPU,
QIM periods, attack family, and PSNR matching. Gaussian/JPEG robustness is
not guaranteed by the conditioning theorem. A 3x3 Gaussian is especially
challenging for a high-pass carrier. This version should not replace v4
before a PSNR-matched evaluation and attack robustness redesign.

## Reproduce

```
PYTHONPATH=src pytest -q tests/test_convolution_qr_v5.py
PYTHONPATH=src python scripts/benchmark_convqr_v5.py --period 24 --repeats 40 \
  --output convqr_v5_benchmark.json
PYTHONPATH=src python scripts/run_main_comparison.py \
  --config configs/experiments/convqr_v5_comparison_experimental.yaml \
  --key bench-convqr-v5 --run-dir runs/convqr-v5
```

Experiment against the 5 main literature baselines using a new experiment
configuration that refers to `configs/methods/proposed_convqr_v5_experimental.yaml`.

**Preliminary result:** with `Delta_u=Delta_v=24`, 6 available classical
512-by-512 hosts and both provided watermarks (12 host-watermark pairs)
produced clean BER=0 and PSNR 50.69--51.41 dB under the test key. This is
not a robustness or multi-seed benchmark. For the benchmark airplane pair,
v5 was faster and used less Python-tracked allocation in both hot paths, but
JPEG90 BER remained weak. Neither algorithm was matched for equal PSNR in the
quick microbenchmark, so that result is not a fair robustness ranking.

**Needed before publication:** 12+ hosts and both watermarks, PSNR-matched
attacks, SSIM, NC, BER distribution, 3 independent runtimes with medians and
peak RSS, ablations (QR-only, Haar-only, coupled, alternate keyed sampling),
error analysis for saturated blocks, and more robust convolutional low-pass
or multi-kernel constraints without host-dependent side information.
