"""New smoke/regression checks, independent of archived V3-V9 tests."""
import numpy as np
import pytest
from pathlib import Path
from qrwatermark.core.factory import build_method
from qrwatermark.core.config import BaselineConfig
from qrwatermark.baselines.direct_qim_control.method import DirectPixelQIM
from qrwatermark.attacks import apply_attack
from qrwatermark.evaluation.review_protocol import split_hosts
from qrwatermark.evaluation.review_statistics import roc_points,host_cluster_ci,calibrate_threshold


def test_direct_qim_control_roundtrip():
    rng=np.random.default_rng(3101)
    host=rng.integers(0,256,(64,64,3),dtype=np.uint8)
    wm=(rng.integers(0,2,(8,8),dtype=np.uint8)*255)
    method=DirectPixelQIM(BaselineConfig(watermark_size=8,quant_step=44,arnold_iterations=0))
    stego=method.embed(host,wm,key=b'fixed-test-key')
    assert stego.side_info is None
    np.testing.assert_array_equal(method.extract(stego.image,key=b'fixed-test-key',watermark_shape=(8,8)).watermark,wm)
    assert np.any(host!=stego.image)


def test_attacks_dont_change_shape():
    host=np.random.default_rng(4).integers(0,256,(64,64,3),dtype=np.uint8)
    for name,options in [('affine_shear',{}),('subpixel_translation',{}),
                         ('aspect_ratio_resample',{}),('green_carrier_null',{})]:
        output=apply_attack(name,host,**options)
        assert output.shape==host.shape and output.dtype==np.uint8
        np.testing.assert_array_equal(output[:,:,0],host[:,:,0]) if name=='green_carrier_null' else None


def test_host_disjoint_split():
    paths=[Path(f'test_{i}.bmp') for i in range(12)]
    a=split_hosts(paths,3101);b=split_hosts(paths[::-1],3101)
    assert a==b
    assert len({str(p) for group in a.values() for p in group})==12
    assert all(len(group)>0 for group in a.values())


def test_bootstrap_unit_is_host_not_attack():
    rows=[{'host':'A','ber':0.0}]*100+[{'host':'B','ber':1.0}]
    result=host_cluster_ci(rows,seed=7,n_boot=500)
    assert result['mean']==0.5
    assert result['n_independent_hosts']==2


def test_roc_does_not_peek_at_test():
    roc,auc=roc_points([0.8,0.9],[0.1,0.2,0.3]);assert auc==1.0
    assert roc.fpr.iloc[0]==0 and roc.tpr.iloc[-1]==1
    assert np.isfinite(calibrate_threshold([0.1,0.3,0.5]))


def test_no_synchronization_or_ecc_new_interface():
    from qrwatermark.proposed.method import GreenQuadWatermark
    from qrwatermark.core.config import ProposedConfig
    m=GreenQuadWatermark(ProposedConfig())
    assert m.config.block_size==4 and m.config.integer_conv4_period==66
    assert m.config.arnold_iterations==0
