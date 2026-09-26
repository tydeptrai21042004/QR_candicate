# Nha et al. (2022) baseline

Reference: P. T. Nha, T. M. Thanh, N. T. Phong, *Consideration of a robust watermarking algorithm for color image using improved QR decomposition*, Soft Computing 26 (2022), 5069–5093, DOI 10.1007/s00500-022-06975-3.

This implementation follows the published 4x4 blue-channel embedding rule: the payload is Arnold-scrambled, embedded in R(1,1) with the paper's quarter-period quantization equations, and extracted from the norm of the received block's first column. The repository benchmark adapts the watermark input to a 64x64 binary watermark so that all methods receive the same payload.
