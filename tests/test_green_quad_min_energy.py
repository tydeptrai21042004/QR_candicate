"""Exact integer minimal-distortion projection and blind extraction tests."""
import numpy as np
import pytest
from qrwatermark.core.config import ProposedConfig
from qrwatermark.proposed.method import GreenQuadWatermark
from qrwatermark.proposed.green_quad_min_energy import optimal_integer_projection


def greedy_reference(x,target):
    z=np.asarray(x,dtype=np.int32).copy()
    while int(z.sum()) != target:
        sign=1 if int(z.sum())<target else -1
        d=z-np.asarray(x,dtype=np.int32)
        costs=np.where((z<255) if sign>0 else (z>0),2*sign*d+1,10**9)
        z[int(np.argmin(costs))]+=sign
    return z

@pytest.mark.parametrize('seed',range(40))
def test_global_l2_projection(seed):
    rng=np.random.default_rng(seed)
    x=rng.choice([0,1,2,10,128,244,253,254,255],size=(12,4)).astype(np.uint8)
    t=rng.integers(0,1021,size=(12,),dtype=np.int32)
    y=optimal_integer_projection(x,t)
    for i in range(len(t)):
        ref=greedy_reference(x[i],int(t[i]))
        assert int(y[i].sum())==int(t[i])
        assert np.sum((y[i].astype('int64')-x[i].astype('int64'))**2)==np.sum((ref.astype('int64')-x[i].astype('int64'))**2)

@pytest.mark.parametrize('seed',range(25))
def test_blind_clean(seed):
    rng=np.random.default_rng(seed)
    host=rng.integers(0,256,size=(64,64,3),dtype=np.uint8)
    wm=rng.integers(0,2,size=(8,8),dtype=np.uint8)*255
    cfg=ProposedConfig(design='green_quad_min_energy',channel=1,integer_conv4_period=66,arnold_iterations=0,watermark_size=8)
    m=GreenQuadWatermark(cfg)
    out=m.embed(host,wm,key=f'key-{seed}'.encode())
    assert out.side_info is None
    recovered=m.extract(out.image,key=f'key-{seed}'.encode(),watermark_shape=(8,8)).watermark
    np.testing.assert_array_equal(recovered,wm)

@pytest.mark.parametrize('value',[0,1,127,128,254,255])
def test_saturated_hosts(value):
    host=np.full((64,64,3),value,dtype=np.uint8)
    wm=np.random.default_rng(77).integers(0,2,size=(8,8),dtype=np.uint8)*255
    cfg=ProposedConfig(design='green_quad_min_energy',channel=1,integer_conv4_period=66,watermark_size=8,arnold_iterations=0)
    m=GreenQuadWatermark(cfg);out=m.embed(host,wm,key=b'extremes')
    np.testing.assert_array_equal(m.extract(out.image,key=b'extremes',watermark_shape=(8,8)).watermark,wm)

def test_reject_wrong_channel():
    with pytest.raises(ValueError):
        GreenQuadWatermark(ProposedConfig(design='green_quad_min_energy',channel=0)).embed(np.zeros((64,64,3),dtype=np.uint8),np.zeros((64,64),dtype=np.uint8),key=b'key')
