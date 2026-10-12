"""Offline regression tests with a FAKE upstream checkout; never reported as official results."""
from __future__ import annotations
import importlib.util, json, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import cv2
import numpy as np
import pytest

SCRIPT=Path(__file__).resolve().parents[1]/'scripts/run_mmwatermark_official.py'
spec=importlib.util.spec_from_file_location('mmw_runner',SCRIPT)
mm=importlib.util.module_from_spec(spec);spec.loader.exec_module(mm)


def mk_repo(tmp_path, fake_source=None):
    path=tmp_path/'upstream'
    (path/'perturbations').mkdir(parents=True)
    (path/'perturbations/image_perturb.py').write_text(fake_source or '''
import numpy as np
from PIL import Image

def gaussian_noise(image,severity):
    assert isinstance(image,np.ndarray)
    arr=np.array(image).copy()
    arr[:,:,0]=np.minimum(255,arr[:,:,0].astype(np.int16)+severity).astype(np.uint8)
    return arr

def jpeg_compression(image,severity=1):
    return np.asarray(image)

def not_official_named_attack(image,severity):
    return image
''')
    subprocess.run(['git','init','-q',str(path)],check=True)
    subprocess.run(['git','-C',str(path),'config','user.email','test@example.test'],check=True)
    subprocess.run(['git','-C',str(path),'config','user.name','Test'],check=True)
    subprocess.run(['git','-C',str(path),'add','.'],check=True)
    subprocess.run(['git','-C',str(path),'commit','-qm','mock source'],check=True)
    subprocess.run(['git','-C',str(path),'remote','add','origin',mm.OFFICIAL_URL],check=True)
    return path


def make_images(tmp_path, count=2):
    p=tmp_path/'images';p.mkdir()
    rng=np.random.default_rng(123)
    for i in range(count):
        im=rng.integers(0,256,size=(128,128,3),dtype=np.uint8)
        assert cv2.imwrite(str(p/f'{i:03}.png'),im)
    return p


def args(tmp_path,**kwargs):
    values=dict(output=tmp_path/'results',official_repo=tmp_path/'upstream',no_clone=True,list_attacks=False,
                images_dir=tmp_path/'images',max_images=2,min_images=1,severities=[1,2],attacks='gaussian_noise',
                strict=True,seed=3101,bootstrap=30,key='test')
    values.update(kwargs)
    return SimpleNamespace(**values)


def test_repository_verified_by_remote_and_sha(tmp_path):
    r=mk_repo(tmp_path)
    assert len(mm.prepare_repo(r,clone=False))==40
    assert mm.find_module(r).name=='image_perturb.py'
    assert mm.discover_names(mm.find_module(r))==['gaussian_noise','jpeg_compression']

@pytest.mark.parametrize('remote', ['https://github.com/attacker/fake.git', 'https://example.com/repo.git'])
def test_wrong_remote_fails_closed(tmp_path,remote):
    r=mk_repo(tmp_path)
    subprocess.run(['git','-C',str(r),'remote','set-url','origin',remote],check=True)
    with pytest.raises(RuntimeError,match='Not the official'):mm.prepare_repo(r,clone=False)


def test_nonempty_unverified_directory_not_cloned(tmp_path):
    p=tmp_path/'wrong';p.mkdir();(p/'existing.txt').write_text('x')
    with pytest.raises(RuntimeError,match='nonempty'):mm.prepare_repo(p)


def test_missing_official_module_fails(tmp_path):
    r=mk_repo(tmp_path);(r/'perturbations/image_perturb.py').unlink()
    with pytest.raises(RuntimeError,match='Expected one'):mm.find_module(r)


def test_valid_official_call_and_rgb_bgr_order(tmp_path):
    r=mk_repo(tmp_path);mod=mm.import_official(mm.find_module(r),r)
    inp=np.zeros((128,128,3),dtype=np.uint8)
    res=mm.call_official(mod.gaussian_noise,inp,2)
    assert res.shape==inp.shape and res.dtype==np.uint8
    assert np.all(res[:,:,2]==2) and np.all(res[:,:,:2]==0)

@pytest.mark.parametrize('bad_code',[
    'def gaussian_noise(image): return image',
    'def gaussian_noise(image,severity,required): return image',
    'def gaussian_noise(image,severity): return image.resize((5,5))',
    'def gaussian_noise(image,severity):\n import numpy as np\n return np.full((128,128,3),np.nan)',
    'def gaussian_noise(image,severity): return "bad"',
])
def test_invalid_official_functions_rejected(tmp_path,bad_code):
    r=mk_repo(tmp_path,bad_code);m=mm.import_official(mm.find_module(r),r)
    with pytest.raises((TypeError,ValueError)):mm.call_official(m.gaussian_noise,np.zeros((128,128,3),np.uint8),1)


def test_normalized_float_official_output(tmp_path):
    r=mk_repo(tmp_path,'def gaussian_noise(image,severity):\n import numpy as np\n return np.ones((128,128,3),dtype=np.float32)*0.5')
    m=mm.import_official(mm.find_module(r),r)
    out=mm.call_official(m.gaussian_noise,np.zeros((128,128,3),np.uint8),1)
    assert np.all(out==128)

@pytest.mark.parametrize('min_size,count,expect',[(128,2,2),(128,1,1),(129,2,0)])
def test_image_size_and_limit(tmp_path,min_size,count,expect):
    img=make_images(tmp_path,count=count)
    if not expect:
        with pytest.raises(RuntimeError):mm.read_images(img,5,min_size=min_size)
    else:assert len(mm.read_images(img,5,min_size=min_size))==expect


def test_image_empty_fails(tmp_path):
    empty=tmp_path/'empty';empty.mkdir()
    with pytest.raises(RuntimeError):mm.read_images(empty,2)

@pytest.mark.parametrize('severity',[0,1,2,3,4,5,6])
def test_severity_cli_range_enforced(tmp_path,severity):
    mk_repo(tmp_path);make_images(tmp_path)
    if severity in range(1,6):
        assert mm.run(args(tmp_path,list_attacks=True,severities=[severity]))['completed']
    else:
        with pytest.raises(ValueError):mm.run(args(tmp_path,severities=[severity]))


def test_only_original_mock_functions_run_and_expected_rows(tmp_path,monkeypatch):
    mk_repo(tmp_path);make_images(tmp_path)
    # The real method is read from the uploaded project, not a synthetic watermark method.
    # Mock source only stands in for upstream until Kaggle Internet is available.
    # Use the watermark assets in project's copied root (created by packaging/test fixture).
    root=Path(__file__).resolve().parents[1]
    (root/'data/watermarks').mkdir(parents=True,exist_ok=True)
    for j in [1,2]:
        f=root/f'data/watermarks/watermark_{j}.png'
        if not f.exists():cv2.imwrite(str(f),np.random.default_rng(j).integers(0,2,size=(64,64),dtype=np.uint8)*255)
    result=mm.run(args(tmp_path,attacks='gaussian_noise,jpeg_compression'))
    assert result['completed'] and result['official_perturbations_executed']
    assert result['expected_rows']==2*2*(1+2*2)
    assert result['actual_rows']==result['expected_rows']
    assert not result['failures']
    assert (tmp_path/'results/attack_summary.csv').exists()


def test_unknown_attack_is_error(tmp_path):
    mk_repo(tmp_path);make_images(tmp_path)
    with pytest.raises(RuntimeError,match='not found'):mm.run(args(tmp_path,attacks='not_an_attack'))
    st=json.loads((tmp_path/'results/run_status.json').read_text())
    assert st['completed'] is False


def test_strict_failure_has_provenance_and_no_false_success(tmp_path):
    mk_repo(tmp_path,'def gaussian_noise(image,severity): raise RuntimeError("unsupported")')
    make_images(tmp_path)
    with pytest.raises(RuntimeError,match='preflight failed'):mm.run(args(tmp_path))
    st=json.loads((tmp_path/'results/run_status.json').read_text())
    assert st['completed'] is False and st['skipped']
    assert st['official_perturbations_executed'] is False


def test_cluster_bootstrap_is_host_based():
    rows=[dict(attack='jpeg',severity=1,host_id='a',nc=.6,ber=.4),
          dict(attack='jpeg',severity=1,host_id='a',nc=.8,ber=.2),
          dict(attack='jpeg',severity=1,host_id='b',nc=1.0,ber=0)]
    s=mm.score_rows(rows,n_boot=100)[0]
    assert s['hosts']==2 and s['observations']==3
    assert s['mean_nc']==pytest.approx(.85)
    assert s['mean_ber']==pytest.approx(.15)
    assert s['nc_ci_low']<=s['mean_nc']<=s['nc_ci_high']


def test_dirty_official_checkout_rejected(tmp_path):
    r=mk_repo(tmp_path)
    (r/'perturbations/image_perturb.py').write_text('def gaussian_noise(image,severity): return image')
    with pytest.raises(RuntimeError,match='uncommitted changes'):
        mm.prepare_repo(r,clone=False)


def test_stochastic_attack_is_seeded_and_rng_state_restored(tmp_path):
    source='''
import numpy as np
from PIL import Image

def gaussian_noise(image,severity):
    a=np.asarray(image).copy()
    a[:,:,0]=(np.random.randint(0,256,size=a.shape[:2])).astype(np.uint8)
    return a
'''
    r=mk_repo(tmp_path,source);module=mm.import_official(mm.find_module(r),r)
    base=np.zeros((128,128,3),np.uint8)
    np.random.seed(123);before=np.random.get_state()
    x=mm.call_official(module.gaussian_noise,base,2,seed=8)
    y=mm.call_official(module.gaussian_noise,base,2,seed=8)
    z=mm.call_official(module.gaussian_noise,base,2,seed=9)
    np.testing.assert_array_equal(x,y)
    assert not np.array_equal(x,z)
    after=np.random.get_state()
    assert before[0]==after[0] and np.array_equal(before[1],after[1])


def test_level_keyword_supported(tmp_path):
    r=mk_repo(tmp_path,'def gaussian_noise(image,level=1): return image')
    module=mm.import_official(mm.find_module(r),r)
    assert mm.call_official(module.gaussian_noise,np.zeros((128,128,3),np.uint8),5).shape==(128,128,3)


def test_missing_independent_images_threshold(tmp_path):
    mk_repo(tmp_path);make_images(tmp_path,count=1)
    with pytest.raises(RuntimeError,match='require 2'):
        mm.run(args(tmp_path,min_images=2))


def test_empty_upstream_attack_file_fails_closed(tmp_path):
    mk_repo(tmp_path,'def something_else(image,severity): return image')
    make_images(tmp_path)
    with pytest.raises(RuntimeError,match='No compatible'):
        mm.run(args(tmp_path,attacks=''))
    st=json.loads((tmp_path/'results/run_status.json').read_text())
    assert st['completed'] is False


def test_non_strict_skips_are_disclosed(tmp_path):
    source='''
from PIL import Image

def gaussian_noise(image,severity): return image

def jpeg_compression(image,severity):
    if severity==2:raise RuntimeError("not available")
    return image
'''
    mk_repo(tmp_path,source);make_images(tmp_path,count=1)
    res=mm.run(args(tmp_path,attacks='gaussian_noise,jpeg_compression',strict=False))
    assert res['completed']
    assert 'jpeg_compression:2' in res['skipped']
    assert res['actual_rows']==1*2*(1+3)


def test_duplicate_severities_rejected(tmp_path):
    mk_repo(tmp_path);make_images(tmp_path)
    with pytest.raises(ValueError,match='unique'):mm.run(args(tmp_path,severities=[1,1]))


def test_repeat_import_does_not_dirty_official_checkout(tmp_path):
    r=mk_repo(tmp_path)
    for _ in range(2):
        mm.import_official(mm.find_module(r),r)
        assert len(mm.prepare_repo(r,clone=False))==40


def test_numpy_array_interface_matches_official_legacy_shape(tmp_path):
    upstream='\nimport numpy as np\ndef fog(x,severity):\n    assert isinstance(x,np.ndarray) and x.dtype==np.uint8\n    h,w=x.shape[:2]\n    return x.copy()\ndef pixelate(x,severity):\n    assert x.ndim==3\n    return x.copy()\n'
    r=mk_repo(tmp_path,upstream)
    mod=mm.import_official(mm.find_module(r),r)
    sample=np.zeros((128,128,3),np.uint8)
    for f in [mod.fog,mod.pixelate]:
        assert mm.call_official(f,sample,1).shape==sample.shape


def test_skimage_multichannel_bridge_matches_channel_axis(tmp_path):
    from skimage.filters import gaussian
    from types import SimpleNamespace
    module=SimpleNamespace(gaussian=gaussian)
    # Modern environments take channel_axis; earlier skimage versions already
    # support the keyword and need no bridge.
    was_adapted=mm.adapt_skimage_gaussian(module)
    data=np.zeros((8,8,3),dtype=np.float64)
    if was_adapted:
        actual=module.gaussian(data,sigma=1,multichannel=True)
        expect=gaussian(data,sigma=1,channel_axis=-1)
        np.testing.assert_allclose(actual,expect)


def test_companion_path_context_restores_path_and_workdir(tmp_path):
    import os,sys
    dependent=tmp_path/'MM_Robustness'; dependent.mkdir()
    before=Path.cwd();paths=list(sys.path)
    def f(x,severity):
        assert Path.cwd()==tmp_path
        assert sys.path[0]==str(tmp_path)
        assert isinstance(x,np.ndarray)
        return x
    result=mm.call_official(f,np.zeros((128,128,3),np.uint8),1,companion_repo=dependent)
    assert result.shape==(128,128,3) and Path.cwd()==before and sys.path==paths
