# APPLY PATCH

Apply these files at the root of your QR_candicate checkout, preserving directories. Overwrite only the two existing runner files. The Green-Quad algorithm is untouched.

After applying:

```bash
python -m pip install -e '.[test]'
python -m pip install imagecorruptions
pytest -q tests/test_existing_hosts_mm_video_fix.py tests/test_mmwatermark_official_integration.py tests/test_video_pipeline_integration.py tests/test_green_quad_min_energy.py
python scripts/run_existing_hosts_mm_video.py --output results/MM_VIDEO --skip-mm --skip-video
```

Run on Kaggle with Internet enabled (official Git clone and optional OpenCV sample video download require it). The root file KAGGLE_ONE_CELL_MMWATERMARK_VIDEO.py can be pasted into one Kaggle notebook cell after committing these files to GitHub.

Original six host images are read from data/hosts/classical; no COCO/Imagenette downloads. The actual official benchmark could not be executed from this offline environment.
