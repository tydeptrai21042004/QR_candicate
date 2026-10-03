# Run the fully blind proposal

The default proposal is now `blind_mecqr_qim_v3`.

## Round trip

```bash
python scripts/run_method.py roundtrip \
  --method proposed \
  --config configs/methods/proposed.yaml \
  --host data/hosts/classical/girl.bmp \
  --watermark data/watermarks/watermark_1.png \
  --key "research-key" \
  --out runs/blind/girl_wm1.png
```

No `.side.npz` is created.

## Extraction from image + key only

```bash
python scripts/run_method.py extract \
  --method proposed \
  --config configs/methods/proposed.yaml \
  --image runs/blind/girl_wm1.png \
  --key "research-key" \
  --out runs/blind/girl_wm1_extracted.png
```

`--watermark` is optional and, when supplied, is used only to print NC/BER after decoding. It is never an input to the decoder.

## Ablations

```bash
python scripts/run_ablation.py \
  --config configs/experiments/ablation.yaml \
  --key "research-key" \
  --run-dir runs/blind_ablation
```

The v3 ablations (`full_blind_v3`, periods, margins, center-QIM, no-certificate, no-Arnold) all have zero side information. The v2/v1 rows are historical non-blind comparisons and are labeled accordingly.
