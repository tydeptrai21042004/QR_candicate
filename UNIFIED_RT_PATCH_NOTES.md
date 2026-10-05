# Unified blind-v3 real-time patch

This patch does **not** create a separate FPGA method. It keeps the existing
`blind_v3` proposal and `configs/methods/proposed.yaml` unchanged, including
`compute_certificate: true` and the same convolution-certificate parameters.

## Mathematical invariants preserved

- one keyed 2x2 carrier per payload bit;
- canonical QR feature `r12 = q1^T b`;
- minimum-energy projection onto the same pixel-feasible safe QIM interval;
- `Delta = 48`, margin ratio `1/8`, 4096-bit payload;
- fully blind extraction and zero serialized side information;
- the same two-extreme convex convolution certificate and path subdivision;
- same Arnold transform and keyed HMAC carrier order.

The speedups are implementation-only:

1. cached key-dependent row/column carrier arrays;
2. cached vectorized Arnold LUTs / direct bit-domain Arnold permutation;
3. closed-form 2x2 canonical-QR `r12` datapath touching only selected pixels;
4. writes only the mathematically modified second-column pixels;
5. exact SSE/changed-sample accounting from selected pixels (no full-frame scan);
6. a repetition=1 specialization of the **same convolution theorem** used by
   blind-v3, removing generic singleton reductions;
7. closed-form handling of zero-first-column QR blocks, eliminating scalar
   `np.linalg.qr` fallbacks on dark/degenerate carriers;
8. Q18 integer reference and ROM-table exporter for FPGA deployment.

## Validation on the bundled data

- `pytest`: 81/81 passed.
- Original vs patched watermarked images: exact SHA-256 equality on all 12
  host/watermark pairs.
- Original vs patched clean extraction: exact equality.
- Original vs patched extraction under all 288 configured standard attack
  instances: exact equality.
- PSNR/SSE/certificate fractions and certificate margins: exact equality in the
  comparison run.
- Q18 reference: 0 modified-pixel mismatches and 0 decoder mismatches across
  the configured standard attacks.
- Universal `proposed.yaml` **with convolution certificate enabled**: every
  bundled pair passed the 100-FPS 10 ms p95 deadline in the local validation;
  worst embedding p95 was about 5.43 ms (~184 FPS) and worst extraction p95
  about 0.45 ms.

Absolute FPS is hardware/software dependent; rerun the supplied validator on
the deployment host/FPGA testbench before reporting a final hardware number.

## Commands

```bash
PYTHONPATH=src pytest -q
PYTHONPATH=src python scripts/run_unified_rt100_validation.py --repeat 60 --target-fps 100
PYTHONPATH=src python scripts/run_fpga_equivalence.py
PYTHONPATH=src python scripts/export_fpga_tables.py --key YOUR_SECRET_KEY
```
