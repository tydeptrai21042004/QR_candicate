# Run the elegant proposal and all ablations

## Full ME-CQR-QIM v2

```bash
pip install -r requirements.txt
pip install -e .

python scripts/run_method.py roundtrip \
  --method proposed \
  --config configs/methods/proposed.yaml \
  --host data/hosts/classical/girl.bmp \
  --watermark data/watermarks/watermark_1.png \
  --key research-key \
  --out runs/elegant/girl_wm1.png
```

## Fast deployment profile

This produces the same watermarked pixels but skips the independent post-embedding certificate audit.

```bash
python scripts/run_method.py roundtrip \
  --method proposed \
  --config configs/methods/proposed_fast.yaml \
  --host data/hosts/classical/girl.bmp \
  --watermark data/watermarks/watermark_1.png \
  --key research-key \
  --out runs/elegant_fast/girl_wm1.png
```

## All v2 ablations + legacy comparison

```bash
python scripts/run_ablation.py \
  --config configs/experiments/ablation.yaml \
  --key research-key \
  --run-dir runs/ablation_elegant_v2
```

Outputs: `ablation.csv`, `ablation_summary.csv`, and `ablation_overall.csv`.

## Tests

```bash
PYTHONPATH=src pytest -q
```

Expected for this package: `69 passed`.
