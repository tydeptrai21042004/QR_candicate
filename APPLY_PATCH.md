# Apply v4 patch

Copy the files in this archive over the same relative paths in the latest corrected repository.

Then run:

```bash
pip install -e .
python -m compileall -q src scripts tests
pytest -q
python scripts/run_hardware_sanity.py --samples 20000
```

Expected final checks:

- `44 passed`;
- `rounded_reconstruction_mismatches=0`;
- `qim_hard_decision_mismatches=0`.

The v4 final path theorem is enabled in `configs/methods/proposed.yaml`. It is applied only after the watermarked image and period allocation are frozen, so it does not change the embedding PSNR or payload.
