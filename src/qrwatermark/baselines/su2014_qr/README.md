# Su et al. 2014 QR baseline

Reference: Q. Su, Y. Niu, G. Wang, S. Jia, J. Yue, *Color image blind watermarking scheme based on QR decomposition*, Signal Processing 94 (2014) 219-235, DOI `10.1016/j.sigpro.2013.06.025`.

The implementation follows the paper's 4x4 QR carrier and its published Eqs. (22)-(28): `r14` is quantized using the two candidate values and extraction uses the parity of `ceil(r14 / Delta)`.

For a controlled benchmark, this repository supplies its common binary payload/Arnold transform and deterministic HMAC block ordering rather than reproducing the paper's original 24-bit RGB watermark serialization and MD5 selector.

**Information model:** blind. The benchmark adaptation does not introduce original-host or original-watermark side information at extraction.
