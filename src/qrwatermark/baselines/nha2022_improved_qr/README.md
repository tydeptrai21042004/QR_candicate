# Nha et al. 2022 improved QR baseline

Reference: P. T. Nha, T. M. Thanh, N. T. Phong, *Consideration of a robust watermarking algorithm for color image using improved QR decomposition*, Soft Computing 26 (2022) 5069-5093, DOI `10.1007/s00500-022-06975-3`.

The implementation preserves the paper's improved QR path rather than calling a generic QR routine:

1. form `M=A.T@A`;
2. solve `R.T@R=M` with the recurrence in Eq. (8);
3. apply the paper's diagonal safeguard when the square-root argument is non-positive;
4. compute `Q` from `A` and the already computed `R` using Eq. (9);
5. embed the bit in `R(1,1)` with the published quarter-period rule;
6. reconstruct `A*=Q R*`;
7. during extraction, obtain `R(1,1)` directly as the Euclidean norm of the first block column (Eq. 12), without recomputing QR.

The baseline is blind: extraction uses the received watermarked image and public method parameters, not the original host or original watermark. The benchmark keeps its common binary payload/Arnold preprocessing, but **does not use the repository HMAC block selector for this method**. Non-overlapping 4x4 blocks are traversed deterministically in raster order, matching the paper procedure.
