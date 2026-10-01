# Audited paper-baseline implementation notes

This repository now contains **six decomposition/QR paper baselines** in the
64x64-payload main comparison, plus one **adaptive-QIM control paper** in a
capacity-compatible 16x16-payload comparison.  The experimental harness
standardizes payload bits, attacks and metrics.  Where a paper used a different
watermark serialization or key/block-selection protocol, that protocol change is
called out explicitly rather than being presented as author-source-exact.

| Code name | Reference | Published carrier reproduced | Default paper value(s) | Reproduction status |
|---|---|---|---|---|
| `su2014_qr` | Su et al., *Signal Processing* 94 (2014) 219-235, DOI `10.1016/j.sigpro.2013.06.025` | 4x4 QR, quantize `r14`, parity extraction (Eqs. 22-28) | `Delta=42` | Equation-level implementation; benchmark payload/key protocol standardized |
| `su2016_hessenberg` | Q. Su, *IET Image Processing* 10(11) (2016) 817-829, DOI `10.1049/iet-ipr.2016.0048` | 4x4 Hessenberg, modify `q22/q32`, absolute-value detector | `T=0.042` | Equation-level implementation with algebraically equivalent deterministic Householder signs |
| `su2017_improved_qr` | Su et al., *Multimedia Tools and Applications* 76(1) (2017) 707-729, DOI `10.1007/s11042-015-3071-x` | 3x3 QR, `q21/q31` relation | repository threshold seed | **Paper-aligned reconstruction**: public text fixes carrier/block structure but not enough implementation detail for a bit-exact author-code claim |
| `su2020_schur` | Su, Zhang & Wang, *Soft Computing* 24 (2020) 445-460, DOI `10.1007/s00500-019-03924-5` | two candidates: Schur-`U` relation and `Dmax` quarter/three-quarter QIM; lower-distortion candidate + mode bit (Eqs. 5-17) | `T=0.03`, `Delta=25` | Equation-level implementation; deterministic Schur-vector sign convention removes numerical +/- ambiguity |
| `chen2021_qqrd` | Chen et al., *Signal Processing* 185 (2021) 108088, DOI `10.1016/j.sigpro.2021.108088` | 4x4 pure-quaternion QR, three `q21/q31` component relations | `T=0.03` initial/reference setting | Published watermark carrier reproduced; direct quaternion-QR reference kernel, **not** the authors' optimized structure-preserving source code |
| `nha2022_improved_qr` | Nha et al., *Soft Computing* 26 (2022) 5069-5093, DOI `10.1007/s00500-022-06975-3` | R-first improved QR, quarter-period `R11`, direct `R11` extraction | `q=10` | Equation-level R-first recurrence and direct extraction implemented |
| `zareian2013_aqim` | Zareian & Tohidypour, *IET Image Processing* 7(5) (2013) 432-441, DOI `10.1049/iet-ipr.2013.0048` | high-entropy 16x16 blocks, 2-level Haar LL2 vector, adaptive QIM and gain-aware detector (Eqs. 1-8) | `Delta0~=0.21`, `gamma~=3.25` for the paper's reported operating point | Equation-level scalar-image method; adapted to one configured color channel |

## Corrections made to the uploaded repository

1. **Su 2014 QR** now uses the paper's `r14` candidate quantizer and its reported
   `Delta=42` operating value instead of a generic QR-QIM approximation.
2. **Su 2016 Hessenberg** replaces the old mislabeled `su2017_hessenberg`
   approximation that embedded in `H`.  The paper actually changes `q22/q32`
   in the orthogonal matrix `Q` using `T=0.042`.  The historical name remains
   only as a compatibility alias.
3. **Su 2020 Schur** now implements both published candidate blocks, compares
   their total squared pixel changes, retains the lower-distortion candidate,
   and stores the one-bit mode flag.  `T=0.03` and `Delta=25` are the paper's
   evaluation settings.  Because a Schur vector can be multiplied by `-1`
   without changing the represented matrix, the implementation fixes a
   deterministic equivalent sign for the carrier pair before applying the
   paper's signed Eq. (16); this prevents LAPACK sign choices from reversing all
   U-mode bits while leaving `A=UDU^T` unchanged.
4. **Nha 2022** no longer calls a generic NumPy QR routine.  It builds `R` first
   from `A^T A = R^T R`, applies the paper's diagonal handling, computes `Q`,
   embeds in `R11`, and extracts `R11` directly from the first block column.
5. **Chen 2021** is treated as a quaternion method rather than three independent
   scalar QR calls.  One selected 4x4 quaternion block carries three bits through
   the i/j/k components of the published `q21/q31` relation.
6. **Zareian 2013 adaptive QIM** is now an actual two-level Haar implementation,
   not a fixed-step QIM label.  Its reported `Delta0` is defined on normalized
   image intensities, so the DWT path works in `[0,1]` and converts back to 8-bit
   pixels only after inverse Haar.  The decoder uses only the published block
   selection map plus `Delta0`, `gamma` and `xi`; no hidden selected-block order
   is passed.
7. **PSNR matching** now tunes each baseline's declared physical strength
   parameter.  For Su 2020, both `T` and `Delta` are scaled jointly.  The CSV
   records the achieved PSNR and residual mismatch instead of claiming an exact
   match when the discrete/rounding rule makes it unattainable.

## Why Zareian 2013 is in a separate comparison profile

The published adaptive-QIM method carries one bit per non-overlapping 16x16 host
block.  A 512x512 host therefore provides only 1024 native carriers, while the
main repository watermark is 64x64 = 4096 bits.  Increasing its capacity by
inventing extra carriers would no longer be the paper baseline.

Therefore:

- `configs/experiments/main_comparison.yaml` uses the native 64x64 proposal
  payload and the six QR/decomposition baselines.
- `configs/experiments/qim_control_comparison.yaml` uses a common 16x16 = 256-bit
  payload across **all seven** paper baselines plus the proposal, so the adaptive
  QIM control remains capacity-valid.

The experiment-level `watermark_size` now overrides each method YAML at runtime,
so the small-payload comparison uses exactly the same bits for every method.

## Reproducibility boundaries

Do **not** describe `su2017_improved_qr` as author-source-exact.  The accessible
paper material verifies the 3x3 block structure, QR decomposition and `q21/q31`
carrier relation, but the authors' original implementation is not publicly
available in this repository.  The module is intentionally labeled a
paper-aligned reconstruction.

Do **not** use the measured runtime of `chen2021_qqrd` as the runtime of Chen et
al.'s optimized structure-preserving QQRD.  This repository reproduces the
quaternion carrier with a clear direct quaternion-QR reference kernel so that
embedding/extraction behavior can be tested independently of unpublished/source-
specific optimization details.

## Validation

Run:

```bash
pytest -q
```

The audit suite checks the Su-2014 `r14` parity quantizer, Su-2016 Hessenberg
reconstruction/update, Su-2020 `Dmax` quarter/three-quarter quantizer and U
candidate, Nha-2022 R-first factorization, quaternion-QR reconstruction,
Zareian-2013 Haar/QIM equations and side-information behavior, factory
registration, and clean embed/extract smoke paths.

Two validation CSVs are included:

- `validation/main64_clean_default_strength.csv`: native 64x64 payload, one real
  Lenna/watermark pair, clean extraction at the paper/default strengths.
- `validation/baseline_audit_comparison.csv`: 16x16 payload, same real pair,
  proposal-PSNR-matched clean/JPEG-50/Gaussian-blur/Gaussian-noise diagnostic
  run including the adaptive-QIM control.

These CSVs are smoke/audit evidence, **not** the final multi-image statistical
results for a paper.  Run the supplied experiment YAMLs over all hosts,
watermarks and attack seeds for manuscript tables.
