# ME-CQR-QIM v2: minimum-energy single-carrier QR-QIM

## Why this revision exists

The previous proposal coupled several mechanisms: repeated carriers, per-group period selection, post-embedding period escalation, and a final certificate tightening pass. The v2 proposal removes those interactions from the embedding rule. Its core is one finite optimization with a closed-form candidate solution.

**No payload bit is repeated.** Each watermark bit changes exactly one 2x2 image block. A keyed candidate pool only gives the encoder several possible locations from which it selects one; every unselected candidate remains untouched.

## Mathematical core

For a 2x2 block

\[
A=[a,b],\qquad q=\frac{a}{\|a\|_2},\qquad s=q^Tb=r_{12}(A),
\]

changing only the canonical QR coefficient `r12` by `delta` gives

\[
A(\delta)=[a,\ b+\delta q].
\]

Because \(\|q\|_2=1\), its continuous embedding energy is exactly

\[
\|A(\delta)-A\|_F^2=\delta^2.
\]

For bit \(m\in\{0,1\}\) and one global QIM period \(\Delta\), define the two cosets

\[
\Lambda_m(\Delta)=\left\{\Delta\left(k+\frac14+\frac m2\right):k\in\mathbb Z\right\}.
\]

The pixel box gives an exact interval \(I_j=[\ell_j,u_j]\) of feasible `r12` displacements for candidate carrier \(j\). For payload bit \(m_i\), the keyed candidate set is \(\mathcal C_i\). The encoder solves

\[
(j_i^\star,\delta_i^\star)
=\arg\min_{j\in\mathcal C_i,\,\delta}\delta^2
\]

subject to

\[
s_j+\delta\in\Lambda_{m_i}(\Delta),\qquad
\delta\in I_j,
\]

and exact uint8 round-trip decodability of the selected block. For each candidate, the feasible lattice index interval is computed directly and the nearest feasible index is obtained by clamping the unconstrained nearest integer. The outer minimization is therefore just an `argmin` over a small candidate dimension and is vectorized for all payload bits.

The default is \(\Delta=64\) and four keyed candidates per bit. Only the selected block is modified.

## Decoder

The authenticated selector stream identifies one selected carrier for each bit. Extraction computes

\[
\hat s_i=r_{12}(\hat A_i),\qquad
\phi_i=\hat s_i\bmod\Delta,
\]

and applies the binary decision

\[
\hat m_i=\mathbf 1\{\phi_i\ge \Delta/2\}.
\]

There is no majority vote, repeated-bit combining, learned decoder, or attack-specific branch.

## Certification is separate from embedding

After the pixels are fixed, the optional convolution theorem is evaluated once. It **cannot change** the carrier, the QIM period, or the watermarked image. If the total certified statistic perturbation bound for bit \(i\) is \(B_i\), then the sufficient condition is

\[
B_i < \frac{\Delta}{4}.
\]

Thus certification is a post-embedding audit of the same simple decoder. `configs/methods/proposed_fast.yaml` disables the audit for deployment/runtime measurements while producing exactly the same watermarked pixels as `proposed.yaml`.

## Complexity and side information

For \(N\) payload bits and candidate-pool size \(M\):

- embedding core: \(O(NM)\), vectorized NumPy arithmetic;
- extraction: \(O(N)\);
- modified carriers: exactly \(N\), independent of \(M\);
- selector overhead: \(N\lceil\log_2 M\rceil\) bits, plus authentication;
- per-bit period codes: **none**;
- repetition: **none**.

For the default 64x64 watermark and \(M=4\), the selector stream is 8192 bits. The old repeated proposal used 8192 period-code bits plus a 4096-bit certificate mask in the logical benchmark accounting.

## Recommended ablations

`configs/experiments/ablation.yaml` contains:

- `full_mecqr`: full v2;
- `pool_1_no_choice`, `pool_2`, `pool_8`: carrier-choice sensitivity without repetition;
- `rounded_sse_selector`: changes only the selection objective;
- `first_feasible_selector`: removes minimum-energy optimization;
- `period_48`, `period_80`: global-margin sensitivity;
- `no_certificate`: verifies that certification affects runtime/analysis only, not pixels or BER;
- `legacy_spread_qim`: exact historical comparison with the former repeated design.
