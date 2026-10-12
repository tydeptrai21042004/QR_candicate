# Minimal Green-Quad benchmark/video patch

Copy **only** the folders/files in this ZIP into the root of your latest `QR_candicate` checkout, preserving their relative paths:

- `scripts/run_mmwatermark_official.py` (NEW)
- `scripts/run_video_pipeline.py` (NEW)
- `tests/test_mmwatermark_official_integration.py` (NEW)
- `tests/test_video_pipeline_integration.py` (NEW)
- `docs/MMWATERMARK_VIDEO_INTEGRATION.md` (NEW)
- `pyproject.toml` (MODIFIED: adds `Pillow>=10.0` only)

No modifications to `src/qrwatermark/proposed/**` or the existing experiments.
No WAVES/OmniSeal adapters are included. No COCO-specific runner is included.

Install: `python -m pip install -e '.[test]'`

Run the test and experiment commands in `docs/MMWATERMARK_VIDEO_INTEGRATION.md`.

**Validation statement**: Tests on a synthetic upstream fixture and synthetic video passed. The real external benchmark requires cloning the upstream repo with working Internet access and may require integration updates for that precise revision. Results are not included or invented.
