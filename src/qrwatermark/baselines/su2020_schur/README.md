# Su–Zhang–Wang 2020 Schur baseline

Reference: Q. Su, X. Zhang, G. Wang, **An improved watermarking algorithm for color image using Schur decomposition**, *Soft Computing* 24 (2020), 445–460. DOI: `10.1007/s00500-019-03924-5`.

This module implements the paper's two-candidate embedding rule on each selected 4×4 scalar color-channel block:

- determine the Schur column `c` associated with the maximum real Schur eigenvalue;
- create candidate `H1*` by modifying `u(2,c),u(3,c)` with threshold `T` (paper Eqs. 7–9);
- create candidate `H2*` by quantizing `Dmax` to `0.25*Delta` or `0.75*Delta` inside the quantization cell (Eqs. 10–11);
- select the candidate with smaller squared pixel change (Eq. 12);
- retain one mode flag per embedded bit (Eq. 13);
- extract by the published U/D rules (Eqs. 16–17).

Paper evaluation values are `T=0.03` and `Delta=25`.  The benchmark can scale both together when PSNR matching is enabled, and records both tuned values plus the residual PSNR mismatch.

The repository deliberately standardizes payload serialization, Arnold scrambling and keyed block selection across all baselines; these protocol choices are not claimed to be the paper authors' source implementation.

Numerical note: Schur vectors are only defined up to a column sign. Because the
paper's Eq. (16) compares signed `u(2,c)` and `u(3,c)` while Eqs. (7)-(8) set
their magnitudes, the implementation fixes an algebraically equivalent carrier-
column sign before both embedding and extraction. This leaves `U D U^T`
unchanged but makes the published detector deterministic across LAPACK builds.
