# Apply v5 edge-runtime patch

Copy the files in this archive over the same relative paths in the current v4 repository. No configuration value, period table, PSNR target, payload size, certificate rule, or public method name is changed.

Then run:

```bash
python -m compileall -q src scripts tests
pytest -q
python scripts/run_hardware_sanity.py --samples 20000 --seed 2026
python scripts/run_edge_benchmark.py --repeat 10 --target-fps 30
```

The optimized code is intended to remain bit-compatible with v4. The bundled regression contains a known-output hash test, plus batch-vs-scalar certificate and decoder equivalence checks.
