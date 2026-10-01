# Su et al. 2017 improved QR baseline

Reference: Q. Su, G. Wang, X. Zhang, G. Lv, B. Chen, *An improved color image watermarking algorithm based on QR decomposition*, Multimedia Tools and Applications 76(1) (2017) 707-729, DOI `10.1007/s11042-015-3071-x`.

The publicly verifiable paper description fixes the key structural details: 3x3 non-overlapping host blocks and watermarking through the relation between `q21` and `q31` of Q, with blind extraction. The authors' implementation is not public. Therefore this module is explicitly labeled a **paper-aligned reconstruction** rather than author-source-exact; it implements the published carrier relation using thresholded relative modulation and the repository's common binary/keying protocol.
