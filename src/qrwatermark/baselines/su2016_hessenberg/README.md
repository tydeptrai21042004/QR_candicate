# Su 2016 Hessenberg baseline

Reference: Q. Su, *Novel blind colour image watermarking technique using Hessenberg decomposition*, IET Image Processing 10(11) (2016) 817-829, DOI `10.1049/iet-ipr.2016.0048`.

The implementation follows the published 4x4 Hessenberg decomposition and Eqs. (11)-(14): watermark bits alter `q22` and `q32` in the orthogonal matrix Q and are decoded by comparing their absolute values. The paper's reported trade-off value `T=0.042` is the default.

This replaces the old repository approximation that incorrectly embedded QIM in a coefficient of H.


Implementation note: Householder-based Hessenberg factors are not unique with respect to paired column/row sign flips. SciPy/LAPACK may return the opposite sign convention from another implementation even though `A = Q H Q^T` is identical. Before applying the paper's `sign(.)` equations, this module uses an equivalent diagonal sign transform `(Q,H) -> (Q D, D H D)` so the result is deterministic while preserving the input block exactly. Extraction still follows Eq. (14), which compares absolute values.

**Information model:** blind. The benchmark keeps key-based block reproducibility but does not provide the original host or watermark to the decoder.
