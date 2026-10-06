# Fully blind coupled QR-QIM v4 patch

This patch adds `blind_cqr_qim_v4` without deleting v3/v2/v1 research paths.

## Mathematical change

For each keyed 2x2 block `A=[a,b]=QR`, the same payload bit is embedded into two observations **inside that same block**:

- `Q` carrier: the direction angle of `q1=a/||a||`.
- normalized `R` carrier:

  `u_R = asinh(r12 / sqrt(r11^2 + r22^2 + 1))`.

If `A` is multiplied by a positive gain, `u_R` is unchanged (up to the fixed `+1` regularizer, negligible away from a degenerate block). The target normalized carrier is converted back in closed form,

`r12* = sqrt(r11^2+r22^2+1) sinh(u_R*)`,

and the image update keeps the original QR core,

`b* = b + (r12* - r12) q1`.

The decoder computes both QIM evidences from the received image and fuses them using only received-block reliability. No original image, selector, period map, certificate mask, reference watermark, or attack label is used.

After the closed-form continuous update, a bounded local integer-lattice closure is applied only when uint8 rounding crosses a QIM decision boundary. This closure stays inside the same 2x2 block, stores no state, and preserves the fully blind decoder.

## Real-time design

- one keyed 2x2 block per payload bit;
- vectorized 4096-block NumPy hot path;
- no per-block `np.linalg.qr`;
- no search over attacks/transforms;
- no side information;
- no repetition into extra image blocks;
- certificate path disabled for v4 online timing.

## Run

Use `configs/methods/proposed.yaml` for v4. Existing v3 tests/configuration remain supported by setting `design: blind_v3`.

For the dedicated ablation:

```bash
python scripts/run_ablation.py \
  --config configs/experiments/ablation_v4.yaml \
  --key paper-key \
  --run-dir runs/blind_v4_ablation
```
