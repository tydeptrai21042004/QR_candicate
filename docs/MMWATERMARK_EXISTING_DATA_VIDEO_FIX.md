# Existing-host MMWatermark and encoded-video evaluation

## Scope

This patch **does not change the Green-Quad watermarking algorithm**: 4096 bits, period 66, no ECC or registration. It fixes a real RGBA input issue and adds an automatic experiment launcher.

- MMWatermark evaluation uses **only the six existing host images** in `data/hosts/classical` and the existing two watermarks. No COCO or Imagenette downloads. The launcher clones the **official** `Jielin-Qiu/MMWatermark-Robustness` checkout automatically (Kaggle Internet must be enabled). It evaluates **only compatible original upstream perturbation functions** discoverable in that checkout, and reports unsupported ones; it does not substitute a different attack suite.
- Transparent black-on-alpha `watermark_1.png` is composited onto **white**, then thresholded, rather than read as grayscale with alpha discarded. Expected pixel counts are **1,407 black and 2,689 white**. Watermark 2 contains **2,037 black and 2,059 white** pixels.
- The video pipeline also uses the correctly composited watermark. It processes a **real** decoded video and scores output after lossless FFV1 and lossy MP4V encode/decode. If no video file is supplied, the launcher attempts to download OpenCV's public `vtest.avi` sample (not synthetic frames) and records the URL and SHA-256. Internet access is required for this optional download.
- A local clean preflight checks all 12 host–watermark pairs, saves expected and recovered PNGs, and verifies clean BER = 0 with correct nonconstant watermark values before the external benchmark begins.

## Quick start after committing the patch files

```bash
python -m pip install -e '.[test]'
python -m pip install imagecorruptions  # optional upstream dependency
python scripts/run_existing_hosts_mm_video.py \
  --output /kaggle/working/MMWATERMARK_VIDEO_RESULTS \
  --min-images 6 --max-images 6 \
  --severities 1 2 3 4 5 \
  --video-frames 300
```

Alternatively, skip video downloading by supplying your own real file:

```bash
python scripts/run_existing_hosts_mm_video.py \
  --video /kaggle/input/myvideos/clip.mp4 \
  --output /kaggle/working/MMWATERMARK_VIDEO_RESULTS
```

If you want to see an explicit failure instead of incomplete coverage whenever *any* official attack/severity is unsupported, add `--strict-mm`. Otherwise the full output records missing attacks and labels the benchmark **PARTIAL**. Either way, if no upstream attacks can execute it is **FAILED**.

For direct MMWatermark execution (official repo cloned if absent):

```bash
python scripts/run_mmwatermark_official.py \
  --images-dir data/hosts/classical \
  --max-images 6 --min-images 6 \
  --severities 1 2 3 4 5 \
  --no-strict --output results/mm_existing_hosts
```

For local clean verification without network access:

```bash
python scripts/run_existing_hosts_mm_video.py \
  --output results/preflight_only --skip-mm --skip-video
```

## Outputs and interpretation

- `preflight/clean_preflight.json`, `preflight/expected_watermark_{1,2}.png`, and `preflight/extracted_watermark_{1,2}.png` show exactly what was embedded/extracted.
- `mmwatermark/run_status.json` contains the official Git commit, source hash, executed attack/severity pairs, skipped attacks, failures, and a `coverage_complete` boolean. `per_sample.csv` and `attack_summary.csv` contain BER/NC if supported upstream calls ran successfully. `mmwatermark/watermark_previews/` contains corrected images.
- `video/FFV1` and `video/mp4v` contain the video and per-frame clean and encoded PSNR, SSIM, NC and BER, and measured FPS. Codec support is checked by OpenCV; unsupported codecs fail with a clear status.
- `execution_status.json` and `MMWATERMARK_VIDEO_RESULTS.zip` collect outcome and records. Encoded videos are excluded from the ZIP to keep it manageable but remain in the output folder.

**Scientific scope:** Using the existing six host images is *not* independent external validation or a reproduction of the original MMWatermark paper's COCO/multimodal evaluation protocol. Describe it as *MMWatermark-Robustness official-source perturbations adapted to the existing six-host dataset*, and report which original functions and severity levels actually executed. The official project delegates some corruptions to its associated `MM_Robustness` dependencies; importing it can fail in environments without those packages. This patch does not invent results or claim guaranteed compatibility with any future upstream revision.

## Verification performed for this patch

- Regression tests using the actual six hosts and both watermarks, including RGBA alpha black/white counts and exact recovered pixels.
- Local synthetic upstream *function fixture* tests for adapter behavior; this is **not** a real official benchmark run.
- Real local FFV1 and MP4V codec integration tests on generated test frames; this tests the pipeline, **not** a published video benchmark.
- No end-to-end official GitHub clone was executable in the authoring environment (network DNS unavailable). Kaggle Internet access is required to obtain this missing verification.

Run the supplied selected tests:

```bash
pytest -q tests/test_existing_hosts_mm_video_fix.py \
          tests/test_mmwatermark_official_integration.py \
          tests/test_video_pipeline_integration.py \
          tests/test_green_quad_min_energy.py
```

The complete legacy test suite has an unrelated `NLMConfig` import error in `tests/test_branch_selector.py`; this patch does not alter legacy APIs to conceal that problem.
