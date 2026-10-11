# Sole proposed method: green, period 66

## Mathematics

For each keyed 4x4 patch, split into four disjoint 2x2 supports. A support with pixels `x` has integer statistic `s=sum(x)` and maximum 1020. With even period T=66, the bit-dependent lattice centres are `T//4 + (T//2)*bit + k*T`, restricted to [0,1020]. Choose the nearest feasible centre `t` and solve `min sum((y_i-x_i)^2)` over `integer 0<=y_i<=255` and `sum(y_i)=t`. This is the exact fixed-centre integer projection, not a novel QR decomposition and not a proof that the chosen lattice centre minimizes over all possible centres.

## Theorem (fixed-centre optimum)

For integer x in [0,255]^4 and feasible sum t, choose an integer dual variable q such that `sum(clip(x+q,0,255)) <= t < sum(clip(x+q+1,0,255))`. Add 1 to `t - sum(clip(x+q,0,255))` eligible coordinates. This attains global minimum squared error for the *fixed sum t*, because exchanging an allocated unit from larger to smaller marginal cost cannot increase the objective. Integer ties may produce multiple minimizers. The production path is vectorized and handles saturation separately.

## Important scientific boundaries

- Blind extraction: image + key + public parameters; no side information.
- Mean attacked BER near 0.24 is not mean NC >=0.90 and does not imply robustness to unknown geometric attacks.
- Mean SSIM near 0.9968 does not satisfy a 0.998 target.
- FPS 175+ must be measured on the exact target machine; avoid claiming earlier local CPU FPS as Kaggle FPS.
- Tests use six unique host files and two watermark files, so they are not 12 distinct images.
- Choose T=66 *before* evaluation. Do not reselect from an ablation using the same test set and claim independent validation.
- Keyed scheduling is not encryption or a defense against an attacker that knows the carrier geometry.
- The `nonoptimal_greedy_control` is a deliberate slow ablation, not another proposed method.
- The minimum-energy proof is conditional on the selected lattice centre; it does not give a probabilistic guarantee under unknown transformations.
