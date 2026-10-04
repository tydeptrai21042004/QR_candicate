# Chen et al. 2021 quaternion QR baseline

Reference: Y. Chen, Z.-G. Jia, Y. Peng, Y.-X. Peng, D. Zhang, *A new structure-preserving quaternion QR decomposition method for color image blind watermarking*, Signal Processing 185 (2021) 108088, DOI `10.1016/j.sigpro.2021.108088`.

The watermark carrier is reproduced from the paper's Eqs. (4)-(6): a 4x4 RGB block is represented by a pure quaternion matrix, and one bit is embedded on each imaginary component by imposing a thresholded relation between quaternion `q21` and `q31`. Therefore the method carries three binary bits per selected block.

The original paper accelerates QQRD with a real structure-preserving algorithm. This repository computes the same mathematical quaternion QR factorization directly with modified Gram-Schmidt and a positive-real R diagonal. It is therefore a reference reimplementation of the watermarking rule, not the authors' optimized factorization source.

**Information model:** blind. The direct quaternion-QR kernel preserves the watermark carrier equations, but its runtime must not be claimed as the authors' optimized structure-preserving QQRD runtime.
