"""Regression tests for alpha-aware watermarking and existing-host auto launcher."""
from __future__ import annotations
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from qrwatermark.core.factory import build_method
from qrwatermark.utils.watermark import prepare_binary_watermark
from qrwatermark.evaluation.metrics import ber,nc

ROOT=Path(__file__).resolve().parents[1]

def load_script(path, name):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj

mm=load_script('run_mmwatermark_official.py','mm_existing_fixed')
video=load_script('run_video_pipeline.py','vid_existing_fixed')
launcher=load_script('run_existing_hosts_mm_video.py','launcher_existing_fixed')


def test_watermark1_is_not_black_after_rgba_preprocess():
    path=ROOT/'data/watermarks/watermark_1.png'
    rgba=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    assert rgba.shape==(64,64,4)
    assert np.all(rgba[:,:,:3]==0)
    prepared=prepare_binary_watermark(path,size=64,background=255)
    assert prepared.shape==(64,64)
    assert set(np.unique(prepared))=={0,255}
    assert np.count_nonzero(prepared==0)==1407
    assert np.count_nonzero(prepared==255)==2689


def test_existing_six_hosts_detected_without_download():
    found=mm.read_images(ROOT/'data/hosts/classical',10)
    assert len(found)==6
    assert len({n for n,_ in found})==6
    assert all(im.ndim==3 and im.shape[2]==3 for _,im in found)


def test_both_watermarks_nonconstant_and_blind_clean_on_existing_hosts():
    method=build_method('proposed')
    hosts=mm.read_images(ROOT/'data/hosts/classical',6)
    for j in (1,2):
        wm=prepare_binary_watermark(ROOT/f'data/watermarks/watermark_{j}.png')
        assert 0<np.count_nonzero(wm)<wm.size
        for _, host in hosts:
            wrapped=method.embed(host,wm,key=b'corrected-mm-video-test')
            recovered=method.extract(wrapped.image,key=b'corrected-mm-video-test',watermark_shape=(64,64)).watermark
            assert wrapped.side_info is None
            assert ber(wm,recovered)==0 and nc(wm,recovered)==pytest.approx(1.0)
            np.testing.assert_array_equal(recovered,wm)


def test_mm_default_images_uses_repository_hosts():
    parsed=mm.parse_args(['--output','/tmp/unused'])
    assert parsed.images_dir==(ROOT/'data/hosts/classical')
    assert parsed.min_images==1


def test_launcher_defaults_six_existing_hosts():
    args=launcher.parse_args([])
    assert args.images_dir is None and args.min_images==6 and args.max_images==6
    assert args.video is None and args.video_frames==300


def test_video_sample_rejects_empty_truncated_file(tmp_path):
    f=tmp_path/'small.avi';f.write_bytes(b'RIFF')
    with pytest.raises(ValueError,match='truncated'):
        launcher.validate_video(f)


def test_video_download_rejects_untrusted_url(tmp_path):
    with pytest.raises(ValueError,match='pinned'):
        launcher.download_video(tmp_path/'out.avi','https://evil.test/data',attempts=1)


def test_launcher_records_mm_error_but_keeps_video(tmp_path,monkeypatch):
    actual=ROOT/'data/hosts/classical'
    sample=tmp_path/'real.avi'
    sample.write_bytes(b'X'*5000)
    monkeypatch.setattr(launcher,'validate_video',lambda p: {'path':str(p),'sha256':'abc'})
    calls=[]
    def runjob(name,cmd,out):
        calls.append(name)
        return {'status':'FAILED' if name=='mmwatermark' else 'PASS','returncode':1 if name=='mmwatermark' else 0}
    monkeypatch.setattr(launcher,'run_command',runjob)
    args=launcher.parse_args(['--output',str(tmp_path/'out'),'--images-dir',str(actual),'--video',str(sample)])
    status=launcher.run(args)
    assert calls==['mmwatermark','video_FFV1','video_mp4v']
    assert status['benchmarks']['mmwatermark']['status']=='FAILED'
    assert status['benchmarks']['video_FFV1']['status']=='PASS'
    assert not status['completed']
    assert (tmp_path/'out/execution_status.json').exists()
    assert (tmp_path/'out/MMWATERMARK_VIDEO_RESULTS.zip').exists()


def test_launcher_video_input_failure_does_not_block_mm(tmp_path,monkeypatch):
    monkeypatch.setattr(launcher,'run_command',lambda name,cmd,out:{'status':'PASS','returncode':0})
    args=launcher.parse_args(['--output',str(tmp_path/'out'),'--video',str(tmp_path/'notthere.avi')])
    status=launcher.run(args)
    assert status['benchmarks']['mmwatermark']['status']=='PASS'
    assert status['benchmarks']['video']['status']=='FAILED'
    assert not status['completed']


def test_mm_runner_alpha_counts_and_preview_with_fake_upstream(tmp_path):
    # Fixture upstream is used for testing the adapter ONLY, never publication results.
    import subprocess
    official=tmp_path/'upstream'
    m=official/'perturbations'/'image_perturb.py'
    m.parent.mkdir(parents=True)
    m.write_text('def gaussian_noise(image,severity): return image\n')
    subprocess.run(['git','init','-q',str(official)],check=True)
    subprocess.run(['git','-C',str(official),'config','user.email','test@test.local'],check=True)
    subprocess.run(['git','-C',str(official),'config','user.name','Test'],check=True)
    subprocess.run(['git','-C',str(official),'add','.'],check=True)
    subprocess.run(['git','-C',str(official),'commit','-qm','fixture'],check=True)
    subprocess.run(['git','-C',str(official),'remote','add','origin',mm.OFFICIAL_URL],check=True)
    args=SimpleNamespace(output=tmp_path/'out',official_repo=official,no_clone=True,list_attacks=False,
        images_dir=ROOT/'data/hosts/classical',max_images=1,min_images=1,severities=[1],
        attacks='gaussian_noise',strict=True,seed=3101,bootstrap=10,key='alpha-case')
    result=mm.run(args)
    assert result['completed'] and result['actual_rows']==4
    assert result['dataset']=='existing_repository_hosts'
    assert result['watermark_bit_counts']['watermark_1.png']['white']==2689
    assert result['watermark_bit_counts']['watermark_1.png']['black']==1407
    rendered=cv2.imread(str(tmp_path/'out/watermark_previews/watermark_1.png'),0)
    assert np.count_nonzero(rendered==255)==2689
    assert result['coverage_complete'] is True
