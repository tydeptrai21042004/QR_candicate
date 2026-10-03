# Fully Blind ME-CQR-QIM v3

## Decoder contract

The v3 decoder requires only:

1. the received watermarked/attacked image,
2. the secret key, and
3. the public method configuration (`watermark_size`, channel, QIM period, Arnold iteration count).

It does **not** use the original host, a selector map, period codes, a certificate mask, a saved `.npz` file, or a reference watermark. `EmbeddingResult.side_info` is exactly `None`.

## Mathematical core

For payload bit `m_i`, the key deterministically assigns one non-overlapping 2x2 carrier. Write the carrier as `A=[a,b]`, define

\[
q_1=\frac{a}{\|a\|_2},\qquad r_{12}=q_1^T b.
\]

Only the second column is changed:

\[
b' = b + \delta q_1.
\]

Hence

\[
r'_{12}=r_{12}+\delta,\qquad \|A'-A\|_F^2=\delta^2.
\]

For period \(\Delta\), the default blind safe sets are

\[
\mathcal S_0(\Delta)=\bigcup_{k\in\mathbb Z}
\Delta[k+1/8,k+3/8],
\]

\[
\mathcal S_1(\Delta)=\bigcup_{k\in\mathbb Z}
\Delta[k+5/8,k+7/8].
\]

The embedder solves one scalar projection:

\[
\delta_i^*=\arg\min_\delta \delta^2
\]

subject to

\[
r_{12,i}+\delta\in\mathcal S_{m_i}(\Delta),
\qquad 0\le b_i+\delta q_{1,i}\le255.
\]

This is a projection onto a periodic union of intervals intersected with the exact pixel-feasible interval. There is no carrier selector and no iterative repair stage.

The hard decoder is simply

\[
\hat m_i = \mathbf 1\{(r_{12,i}\bmod\Delta)\ge\Delta/2\}.
\]

With the default margin ratio `1/8`, every continuous codeword is at least \(\Delta/8\) from a hard-decision boundary. Because only two pixels in the second column are rounded, the induced feature error is bounded by

\[
|q_1^T e|\le\|e\|_2\le\sqrt{2}/2.
\]

Thus clean uint8 decoding is protected whenever \(\Delta/8>\sqrt{2}/2\), which is easily satisfied by the default \(\Delta=48\).

## Side-information accounting

For v3:

- selector bits: **0**
- period-code bits: **0**
- certificate-mask bits: **0**
- side-info authentication bits: **0**
- serialized side-information bits: **0**

The optional certificate is a report-only post-embedding audit and is never serialized or consumed by extraction.

## Run

Embed:

```bash
python scripts/run_method.py embed \
  --method proposed \
  --config configs/methods/proposed.yaml \
  --host data/hosts/classical/girl.bmp \
  --watermark data/watermarks/watermark_1.png \
  --key "research-key" \
  --out runs/blind/girl.png
```

Blind extraction — note that there is no `--side-info` and no reference watermark:

```bash
python scripts/run_method.py extract \
  --method proposed \
  --config configs/methods/proposed.yaml \
  --image runs/blind/girl.png \
  --key "research-key" \
  --out runs/blind/girl_extracted.png
```

Run all v3 ablations:

```bash
python scripts/run_ablation.py \
  --config configs/experiments/ablation.yaml \
  --key "research-key" \
  --run-dir runs/blind_ablation
```
