# Nha et al. 2022 improved QR baseline

Reference: P. T. Nha, T. M. Thanh, N. T. Phong, *Consideration of a robust watermarking algorithm for color image using improved QR decomposition*, Soft Computing 26 (2022) 5069-5093, DOI `10.1007/s00500-022-06975-3`.

This implementation uses the paper's **R-first** factorization, not a generic NumPy QR call:

1. form `M=A.T@A`;
2. solve `R.T@R=M` with the recurrence in Eq. (8);
3. set `r_ii=1` for `i>=2` if the square-root argument is non-positive, as specified in the paper;
4. compute Q from A and R using Eq. (9);
5. embed in `R(1,1)` with the published quarter-period rule;
6. extract `R(1,1)` directly as the Euclidean norm of the first block column (Eq. 12), avoiding QR during decoding.
