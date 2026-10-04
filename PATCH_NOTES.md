# Paper-baseline fidelity patch

This patch removes the unverifiable Su-2017 baseline and keeps each remaining literature baseline faithful to its published carrier equations and decoder information model while adapting only the surrounding benchmark protocol (common host loading, binary payload, attacks and metrics).

## Active baselines

- `su2014_qr` — blind; 4x4 QR, `R(1,4)` paper quantizer, `Delta=42`.
- `su2016_hessenberg` — blind; 4x4 Hessenberg, `Q(2,2)/Q(3,2)`, `T=0.042`.
- `su2020_schur` — retains the published per-block embedding-mode flag; both Schur candidates, minimum-distortion selection, `T=0.03`, `Delta=25`.
- `chen2021_qqrd` — blind; whole-color quaternion 4x4 QR and the three published `q21/q31` imaginary-component relations.
- `nha2022_improved_qr` — blind; paper's blue-channel 4x4 raster traversal, R-first factorization, `R(1,1)` QIM, `q=10`, and direct extraction from the first column norm.
- `zareian2013_aqim` — side-information-assisted/semi-blind; preserves the selected entropy-block map, `Delta0`, `gamma`, `xi`, two-level Haar adaptive QIM, gain estimation, and the published `[-10,10]` degree / `0.5` degree rotation search.

## Removed

`su2017_improved_qr` is removed because the exact published numerical embedding rule was not sufficiently verifiable for a defensible reproduction. The historical `su2017_hessenberg` compatibility alias is also removed because it was not a separate faithful paper baseline.

The previous baseline-audit CSVs are removed as stale because they contain Su-2017 and/or were generated before the fidelity corrections. Re-run experiments to produce new result tables.

## Validation

```bash
pytest -q
```

Current repository result: **77 passed**.

The baseline-specific tests additionally check required side information, Nha-2022 paper block order, removal of Su-2017 factory registrations, and Zareian-2013 entropy-map synchronization on an unknown +5 degree rotation.

See `docs/PAPER_BASELINES.md` for the exact paper-vs-pipeline boundary for every method.
