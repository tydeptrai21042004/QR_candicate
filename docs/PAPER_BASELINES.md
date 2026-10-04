# Paper-baseline fidelity audit

The benchmark keeps the repository's common host loading, binary payload format,
attack suite and metrics, but it does **not** change a paper's extraction-information
model. A blind paper stays blind; a method that transmits decoder side information
keeps that side information. Changes to payload serialization or keyed block ordering
are documented as benchmark adaptations rather than presented as author-source-exact.

`su2017_improved_qr` has been removed because the available paper material was not
sufficient to defend the repository's threshold rule as a faithful reproduction. The
old `su2017_hessenberg` compatibility alias has also been removed because it was a
historical mislabeled alias, not an independent Su-2017 paper baseline.

| Code name | Reference | Paper mechanism retained | Information model retained | Benchmark adaptation / boundary |
|---|---|---|---|---|
| `su2014_qr` | Su et al., *Signal Processing* 94 (2014), 219-235, DOI `10.1016/j.sigpro.2013.06.025` | non-overlapping 4x4 QR; `r14` candidate quantizer; parity extraction; `Delta=42` | **Blind**: extraction needs no original host/watermark; block selection is reproduced from the benchmark key | common binary payload and deterministic keyed selection replace the paper's 24-bit RGB serialization / MD5 selector |
| `su2016_hessenberg` | Q. Su, *IET Image Processing* 10(11) (2016), 817-829, DOI `10.1049/iet-ipr.2016.0048` | 4x4 Hessenberg; modify `q22,q32`; Eq. (14) absolute-value detector; `T=0.042` | **Blind** | common binary payload and benchmark keyed block order replace the paper's color-watermark serialization / MD5 selector; deterministic equivalent Hessenberg signs only remove LAPACK ambiguity |
| `su2020_schur` | Su, Zhang & Wang, *Soft Computing* 24 (2020), 445-460, DOI `10.1007/s00500-019-03924-5` | Schur-`U` and `Dmax` candidates; lower-distortion choice; `T=0.03`, `Delta=25` | **Side-information-assisted / semi-blind in this benchmark taxonomy**: the paper's one mode flag per embedded bit is retained and required for extraction | common binary payload and keyed block ordering; deterministic equivalent Schur-vector signs remove numerical +/- ambiguity |
| `chen2021_qqrd` | Chen et al., *Signal Processing* 185 (2021), 108088, DOI `10.1016/j.sigpro.2021.108088` | 4x4 pure-quaternion block; three imaginary `q21/q31` relations; `T=0.03` | **Blind** | common binary payload / keyed block order; direct quaternion QR reproduces the carrier mathematics but is **not** the authors' optimized structure-preserving runtime kernel |
| `nha2022_improved_qr` | Nha, Thanh & Phong, *Soft Computing* 26 (2022), 5069-5093, DOI `10.1007/s00500-022-06975-3` | R-first factorization from `A^T A`; quarter-period `R11` rule; direct first-column-norm extraction; `q=10` | **Blind** | common binary payload; importantly, the implementation now uses the paper's deterministic non-overlapping block scan rather than the repository key selector |
| `zareian2013_aqim` | Zareian & Tohidypour, *IET Image Processing* 7(5) (2013), 432-441, DOI `10.1049/iet-ipr.2013.0048` | high-entropy 16x16 blocks; two-level Haar LL2; adaptive QIM; gain-aware detector; entropy-map rotation search over `[-10,10]` degrees in `0.5` degree steps | **Semi-blind**: selected-block map plus `Delta0`, `gamma`, and `xi` are retained and required | scalar paper method is applied to the configured benchmark color channel; native one-bit-per-16x16-block capacity is preserved |

## Fidelity rules used in the code

1. **Carrier equations are paper equations.** No generic QIM/QR surrogate is used under a paper name.
2. **Information assumptions are not upgraded.** `Su2020Schur` still requires its mode flags; `Zareian2013AdaptiveQIM` still requires its published side information. They are not relabeled or modified into fully blind methods.
3. **Native capacity is not inflated.** Zareian 2013 remains one bit per 16x16 selected block, so it belongs in the small-payload QIM-control profile rather than the 4096-bit main comparison.
4. **Pipeline adaptations are external to the carrier.** Common payload serialization, host loading, attacks, metrics and (where the paper already used secret block selection) repository keying may be standardized. Such runs should be described as *protocol-adapted equation-level reproductions*.
5. **Paper operating points and PSNR-matched operating points are distinct.** Strength matching is useful for a fair distortion-controlled study, but a tuned strength is not the paper's native default result.

## Method-specific notes

### Su 2014 QR

The implementation uses the two candidate values from Eqs. (22)-(26) for `r14`
and decodes with the parity of `ceil(r14/Delta)` as in Eq. (28). The default is
`Delta=42`. The paper is explicitly blind.

### Su 2016 Hessenberg

The watermark is embedded in `q22` and `q32` of the orthogonal Hessenberg factor,
not in the Hessenberg matrix `H`. Extraction compares the published absolute-value
relation. A paired diagonal sign transform is used only to make the mathematically
non-unique Householder signs deterministic while preserving `A = Q H Q^T` exactly.

### Su 2020 Schur

Both published candidates are computed. The lower squared-pixel-distortion candidate
is selected and the corresponding `U`/`D` mode flag is stored. Extraction refuses to
run without those flags. This intentionally preserves the paper's external flag file.

### Chen 2021 QQRD

The RGB block is represented as a pure quaternion matrix, and three bits are embedded
through the `i`, `j`, and `k` components of the published `q21/q31` relation. The
factorization here is a direct quaternion modified-Gram-Schmidt reference. Therefore
PSNR/NC behavior can be compared at equation level, but its measured runtime must not
be reported as the runtime of Chen et al.'s optimized structure-preserving QQRD.

### Nha 2022 improved QR

The implementation builds `R` first from `A^T A = R^T R`, then obtains `Q`, embeds in
`R(1,1)`, and extracts that coefficient directly from the first block-column norm.
The previous repository HMAC/keyed block selector has been removed for this baseline;
blocks are now traversed deterministically in raster order as in the paper procedure.

### Zareian 2013 adaptive QIM

The selected high-entropy-block map and `Delta0`, `gamma`, `xi` are transmitted as
side information. During extraction the received image is inverse-rotated over the
paper's candidate grid, the high-entropy blocks are recomputed, and the candidate with
the largest overlap with the transmitted original block map is selected before gain
estimation and QIM decoding. This keeps the method semi-blind rather than silently
turning it into a blind baseline.

## Comparison profiles

- `configs/experiments/main_comparison.yaml`: proposal + the five decomposition/QR
  baselines that can support the main 64x64 (4096-bit) payload.
- `configs/experiments/qim_control_comparison.yaml`: proposal + the same five
  decomposition/QR baselines + Zareian 2013 at a common 16x16 (256-bit) payload.

## Validation

Run:

```bash
pytest -q
```

The tests check the published Su-2014 quantizer, Su-2016 Hessenberg relation,
Su-2020 two-candidate rules and required flags, Chen quaternion QR reconstruction,
Nha R-first reconstruction/raster block order, Zareian Haar/QIM equations and
semi-blind side information, plus Zareian's entropy-map registration under an
unknown +5 degree rotation.
