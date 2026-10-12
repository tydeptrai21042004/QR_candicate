# Green-Quad: MMWatermark-Robustness official-source perturbations + actual video pipeline

## Scope and scientific claims

- Only Green-Quad (`build_method('proposed')`) is evaluated. The embedding algorithm is **not changed**.
- The image benchmark clones the official project `https://github.com/Jielin-Qiu/MMWatermark-Robustness.git`, checks the exact upstream commit and Git remote, and discovers compatible named functions defined in its `perturbations/**/image_perturb.py`. The runner never substitutes a locally implemented attack.
- **This is not necessarily a full reproduction of the original publication.** The paper uses specific generated images / COCO-based procedures and additional tools; testing new independent images with functions from upstream is an *official-source perturbation evaluation*. Call it a "MMWatermark-Robustness perturbation protocol adaptation" unless its full original dataset and attack coverage are reproduced.
- Upstream may call external dependency code (notably `MM_Robustness`) and some functions may have incompatible signatures. The runner is fail-closed by default and lists skipped functions. It does not download additional models implicitly or fabricate results.
- **No actual upstream clone was executable in the artifact-building environment** (outbound GitHub DNS unavailable). Unit tests use a clearly labelled synthetic upstream fixture. Run on an Internet-enabled Kaggle runtime to establish real results.
- The official repository's license is CC BY-NC-SA; check the applicable terms for your use and attribution.

## Requirements

Python >=3.10 with this project's editable dependencies. The patch also adds Pillow to `pyproject.toml`. Git and access to github.com are required for first clone. Never store your benchmark outputs inside the upstream Git checkout.

## 1. Inspect official attack coverage

```bash
python scripts/run_mmwatermark_official.py \
  --official-repo third_party/MMWatermark-Robustness \
  --list-attacks \
  --output results/mmwatermark
```

This clones the official upstream repo if absent and prints the **source-defined** matching attack names. If the official file does not define the expected functions, it prints an empty list. That is a real incompatibility, not something to hide or automatically replace. Check `run_status.json`, install dependencies from the official project's instructions or write an explicit audited adapter after inspecting the exact commit.

## 2. Run external-host image robustness evaluation

Provide independent >=128x128 pixel 3-channel images (not the same six development/test hosts). Use:

```bash
python scripts/run_mmwatermark_official.py \
  --images-dir /kaggle/input/independent-watermark-hosts/images \
  --max-images 100 --min-images 100 \
  --severities 1 2 3 4 5 \
  --official-repo third_party/MMWatermark-Robustness \
  --output /kaggle/working/MMWATERMARK_RESULTS
```

`--strict` is ON by default: any incompatible discovered attack-severity pair fails before running, and mid-run failures abort rather than silently omitting difficult attacks. For exploratory mode only, `--no-strict` writes explicit coverage omissions in `run_status.json`. You may restrict to known discovered names using `--attacks gaussian_noise,jpeg_compression`, but report the subset rather than claiming all benchmark attacks.

Outputs:
- `run_status.json`: upstream URL, SHA, imported file hash, discovered/executed/skipped functions, completion, expected vs actual observations.
- `per_sample.csv`: image ID, watermark ID, attack, severity, clean PSNR/SSIM, attacked NC/BER.
- `attack_summary.csv`: host-cluster means and 95% bootstrap intervals, with number of independent image clusters.

Only report benchmark results when `completed=true`, `official_perturbations_executed=true` and `expected_rows=actual_rows`. Report any skipped attack families prominently.

## 3. Run an actual video encode/decode benchmark

```bash
# Lossless, verifies clean recovery after a genuine codec cycle
python scripts/run_video_pipeline.py \
  --input-video /kaggle/input/independent-videos/test.mp4 \
  --output-video /kaggle/working/watermarked_ffv1.mkv \
  --results-dir /kaggle/working/VIDEO_FFV1 \
  --codec FFV1 --max-frames 300

# Lossy; codec may destroy watermark bits, which is measured rather than hidden
python scripts/run_video_pipeline.py \
  --input-video /kaggle/input/independent-videos/test.mp4 \
  --output-video /kaggle/working/watermarked_mp4v.mp4 \
  --results-dir /kaggle/working/VIDEO_MP4V \
  --codec mp4v --max-frames 300
```

The input video must have stable dimensions and sufficient 4x4 blocks for 4096 payload bits. The video pipeline does not rescale input images or silently add ECC, spatial synchronization, or watermark repetition. Watermark `data/watermarks/watermark_1.png` is embedded in every decoded frame. Output video does not retain audio. On some systems, OpenCV FFV1 is unavailable; if opening that codec fails, treat the experiment as unsupported, not as a completed test.

Outputs:
- `video_summary.json`: true decoded/encoded frame counts, source fps, combined embed/extract timings, decode-embed-encode fps, post-codec NC/BER, PSNR/SSIM, and whole run FPS including verification.
- `video_clean_frames.csv`: per-frame in-memory fidelity and exact clean recovery.
- `video_encoded_frames.csv`: per-frame post-codec watermark recovery and fidelity.

The existing ~603 FPS figure is a **memory-only still-image benchmark** and must not be renamed "video FPS". The new `total_e2e_fps` and `decode_embed_encode_fps` have different denominators; include the definitions in the manuscript.

## 4. Automated tests

```bash
python -m pytest -q tests/test_mmwatermark_official_integration.py tests/test_video_pipeline_integration.py
python -m pytest -q tests/test_green_quad_min_energy.py tests/test_final_single_proposal.py tests/test_metrics.py tests/test_qr_complete_runner_v3.py
```

The newly added upstream-module fixture is synthetic and deliberately not reported as an official benchmark. Cases include: official clone provenance, dirty/unverified checkout, incompatible signatures, image geometry and dtype validation, deterministic stochastic perturbations, fail-closed attack coverage, bootstrap cluster accounting, minimum independent-host count, invalid attack lists, FFV1 and mp4v codec loops, frame limits, missing input, audio-free output, corrupted/small frame and FPS fields.

Some existing legacy tests in the uploaded repo import the removed `NLMConfig`; running the complete old suite can fail at collection for reasons unrelated to these two new runners. This patch does not modify the legacy watermark proposal or compatibility aliases.
