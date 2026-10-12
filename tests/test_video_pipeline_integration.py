from __future__ import annotations
import importlib.util,json,os
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest
SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_video_pipeline.py'
spec=importlib.util.spec_from_file_location('video_runner',SCRIPT)
vid=importlib.util.module_from_spec(spec);spec.loader.exec_module(vid)


def mk_video(path,count=4,size=(128,128),fourcc='FFV1'):
    w=cv2.VideoWriter(str(path),cv2.VideoWriter_fourcc(*fourcc),12,size)
    if not w.isOpened():pytest.skip('Local OpenCV does not support this test video codec')
    rng=np.random.default_rng(9)
    for i in range(count):w.write(rng.integers(0,256,size=(size[1],size[0],3),dtype=np.uint8))
    w.release()


def a(tmp_path,**kw):
    d=dict(input_video=tmp_path/'input.mkv',output_video=tmp_path/'output.mkv',
           results_dir=tmp_path/'results',codec='FFV1',max_frames=4,fallback_fps=30.0,key='test-video')
    d.update(kw);return SimpleNamespace(**d)


def test_lossless_video_roundtrip(tmp_path):
    mk_video(tmp_path/'input.mkv')
    res=vid.run(a(tmp_path))
    assert res['completed'] and res['decoded_output_frames']==4
    assert res['input_frames_processed']==4
    assert res['encoded_ber']['max']==0
    assert res['clean_psnr']['min']>0
    assert res['total_e2e_fps']>0 and res['decode_embed_encode_fps']>0
    assert (tmp_path/'results/video_clean_frames.csv').exists()


def test_lossy_video_is_reported_without_false_clean_claim(tmp_path):
    mk_video(tmp_path/'input.mkv')
    r=vid.run(a(tmp_path,codec='mp4v',output_video=tmp_path/'output.mp4',max_frames=3))
    assert r['completed'] and r['decoded_output_frames']==3
    assert 0<=r['encoded_ber']['mean']<=1 and 0<=r['encoded_nc']['mean']<=1


def test_frame_limit(tmp_path):
    mk_video(tmp_path/'input.mkv',count=5)
    r=vid.run(a(tmp_path,max_frames=2))
    assert r['decoded_output_frames']==2


def test_too_small_frame_is_explicit_failure(tmp_path):
    mk_video(tmp_path/'input.mkv',size=(64,64))
    with pytest.raises(RuntimeError,match='too small'):vid.run(a(tmp_path))
    info=json.loads((tmp_path/'results/video_summary.json').read_text())
    assert not info['completed'] and 'too small' in info['failure']


def test_missing_video(tmp_path):
    with pytest.raises(FileNotFoundError):vid.run(a(tmp_path))


def test_prevent_input_output_overwrite(tmp_path):
    mk_video(tmp_path/'input.mkv')
    with pytest.raises(ValueError,match='distinct'):vid.run(a(tmp_path,output_video=tmp_path/'input.mkv'))


def test_zero_frame_limit(tmp_path):
    mk_video(tmp_path/'input.mkv')
    with pytest.raises(ValueError,match='positive'):vid.run(a(tmp_path,max_frames=0))


def test_invalid_codec(tmp_path):
    with pytest.raises(ValueError):vid.writer_for(tmp_path/'bad.mkv','X',25,(128,128))

@pytest.mark.parametrize('n',[1,2,5,25])
def test_latency_summary_statistics(n):
    summary=vid.summarize([2.0]*n)
    assert summary['median']==2.0 and summary['p95']==2.0
    assert vid.summarize([]) is None


def test_lossless_post_codec_fidelity(tmp_path):
    mk_video(tmp_path/'input.mkv',count=2)
    result=vid.run(a(tmp_path,max_frames=2))
    assert result['encoded_psnr']['mean']>0
    assert result['encoded_ssim']['mean']<=1
    assert result['encoded_ber']['max']==0


def test_lossy_post_codec_fidelity(tmp_path):
    mk_video(tmp_path/'input.mkv',count=2)
    result=vid.run(a(tmp_path,max_frames=2,codec='mp4v',output_video=tmp_path/'out.mp4'))
    assert result['encoded_psnr']['min']>0 and 0<=result['encoded_ssim']['mean']<=1


@pytest.mark.parametrize('wm_idx',[1,2])
def test_video_both_watermarks_lossless_with_separate_outputs(tmp_path,wm_idx):
    mk_video(tmp_path/'input.mkv',count=3)
    path=SCRIPT.parents[1]/'data/watermarks'/f'watermark_{wm_idx}.png'
    args=a(tmp_path,watermark_path=path,max_frames=3,
           results_dir=tmp_path/f'watermark_{wm_idx}')
    res=vid.run(args)
    assert res['completed'] and res['watermark_file']==path.name
    assert res['encoded_ber']['max']==0
    assert res['encoded_nc']['min']==pytest.approx(1.0)
    assert res['algorithm_only_embed_extract_fps_from_mean_ms']>0
    assert res['codec_inclusive_stage_fps_estimate']>0
    assert (args.results_dir/'expected_watermark.png').is_file()
