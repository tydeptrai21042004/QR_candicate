# Fully Blind v3 change log

- Added `design: blind_v3` as the default proposal.
- Removed all per-image decoder side information from v3.
- Replaced content-dependent carrier selection by one key-determined carrier per payload bit.
- Replaced point-center QIM by minimum-energy projection onto a symmetric safe decision interval.
- Kept payload multiplicity exactly one; no repetition, voting, ECC, or hidden duplicate payload is used.
- Added a report-only blind certificate based on the actual rounded decision margin; no certificate mask is serialized.
- Updated CLI extraction so `--side-info` and `--watermark` are unnecessary for v3.
- Added blind-specific ablations for period, safe margin, center-QIM, certificate, and Arnold scrambling.
- Preserved selector-based v2 and repeated v1 as explicit historical configurations/ablations.
- Added v3 regression tests and validation CSVs.
