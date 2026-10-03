# Elegant v2 change log

- Replaced the default repeated spread-QIM proposal with ME-CQR-QIM v2.
- Every payload bit is embedded exactly once in one 2x2 block.
- Removed per-bit adaptive period selection from v2; one global QIM period is used.
- Replaced rule stacking with a single minimum-energy finite optimization over a keyed carrier-choice set.
- Vectorized candidate QR geometry, QIM projection, rounding validation, selection, and decoding.
- Made certification a single independent post-embedding audit; it cannot alter pixels.
- Added compact authenticated carrier-selector side information.
- Added a fast profile with certification disabled but bit-identical embedding output.
- Preserved the former repeated proposal as `legacy_spread_qim` for direct ablation.
- Expanded the ablation suite to 10 variants.
- Added tests for no repetition, selector overhead, and certificate/pixel independence.
