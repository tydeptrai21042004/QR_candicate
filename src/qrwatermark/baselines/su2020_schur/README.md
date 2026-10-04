# Su–Zhang–Wang 2020 Schur baseline

Reference: Q. Su, X. Zhang, G. Wang, **An improved watermarking algorithm for color image using Schur decomposition**, *Soft Computing* 24 (2020), 445-460. DOI: `10.1007/s00500-019-03924-5`.

This module implements the paper's two-candidate embedding rule on each selected 4x4 scalar color-channel block:

- determine the Schur column `c` associated with `Dmax`;
- create candidate `H1*` by modifying `u(2,c),u(3,c)` with threshold `T` (Eqs. 7-9);
- create candidate `H2*` by quantizing `Dmax` to `0.25*Delta` or `0.75*Delta` in its quantization cell (Eqs. 10-11);
- select the candidate with smaller squared pixel change (Eq. 12);
- **retain one mode flag per embedded bit** (Eq. 13);
- extract with the published U/D rules (Eqs. 16-17).

Paper evaluation values are `T=0.03` and `Delta=25`.

The external mode flags are not discarded. In this benchmark taxonomy the method is therefore treated as **side-information-assisted / semi-blind**, even though it does not need the original host image or original watermark. Extraction raises an error if the flag array is absent.

The repository standardizes binary payload serialization and keyed block ordering. These are benchmark protocol adaptations, not claims of author-source identity.

Numerical note: real Schur vectors have an arbitrary column sign. The code applies an algebraically equivalent sign convention before embedding/extraction so the signed published detector is deterministic across LAPACK implementations while preserving `A=U D U^T`.
