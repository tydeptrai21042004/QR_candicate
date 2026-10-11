import numpy as np
import pytest
from qrwatermark.core.config import ProposedConfig
from qrwatermark.core.factory import build_method
from qrwatermark.proposed.method import GreenQuadWatermark
from qrwatermark.proposed.green_quad_min_energy import optimal_integer_projection,greedy_integer_projection

def test_only_one_proposal_registered():
    a=build_method('proposed','configs/methods/proposed.yaml')
    assert isinstance(a,GreenQuadWatermark)
    assert a.config.channel==1 and a.config.integer_conv4_period==66 and a.config.arnold_iterations==0
    with pytest.raises(ValueError):build_method('integer_conv_quad_v9')
    with pytest.raises(ValueError):ProposedConfig(design='integer_conv_quad_v9').validate()

def test_ablation_mode_guards():
    with pytest.raises(ValueError):ProposedConfig(channel=0).validate()
    with pytest.raises(ValueError):ProposedConfig(arnold_iterations=10).validate()
    with pytest.raises(ValueError):ProposedConfig(projection='greedy_control').validate()
    ProposedConfig(channel=0,ablation_mode=True).validate()
    ProposedConfig(arnold_iterations=10,ablation_mode=True).validate()

def test_integer_energy_is_better_than_greedy_control():
    x=np.asarray([[0,1,254,255],[13,44,240,250],[255,255,0,0]],dtype=np.uint8)
    t=np.asarray([550,503,550],dtype=np.int32)
    o=optimal_integer_projection(x,t).astype(np.int32)
    g=greedy_integer_projection(x,t).astype(np.int32)
    for v in (o,g):np.testing.assert_array_equal(v.sum(axis=1),t)
    cost_opt=((o-x.astype(np.int32))**2).sum(axis=1)
    cost_greedy=((g-x.astype(np.int32))**2).sum(axis=1)
    assert np.all(cost_opt<=cost_greedy)
    assert np.any(cost_opt<cost_greedy)

def test_rgb_other_channels_untouched():
    rng=np.random.default_rng(2026)
    host=rng.integers(0,256,size=(64,64,3),dtype=np.uint8)
    wm=(rng.integers(0,2,size=(8,8),dtype=np.uint8)*255)
    method=GreenQuadWatermark(ProposedConfig(watermark_size=8))
    out=method.embed(host,wm,key=b'key')
    np.testing.assert_array_equal(out.image[:,:,[0,2]],host[:,:,[0,2]])
    np.testing.assert_array_equal(method.extract(out.image,key=b'key',watermark_shape=wm.shape).watermark,wm)


def test_block_mapper_key_stability_and_geometry():
    from qrwatermark.proposed.block_carrier import keyed_blocks
    rr,cc=keyed_blocks((512,512),1024,b'key1')
    r2,c2=keyed_blocks((512,512),1024,b'key1')
    assert rr is r2 and cc is c2
    assert len(set(zip(rr.tolist(),cc.tolist())))==1024
    assert (rr%4==0).all() and (cc%4==0).all()
    other,_=keyed_blocks((512,512),1024,b'key2')
    assert not np.array_equal(rr,other)
    with pytest.raises(ValueError):rr[0]=8


def test_wrong_key_and_audit_metadata():
    rng=np.random.default_rng(444)
    host=rng.integers(0,256,size=(64,64,3),dtype=np.uint8)
    wm=rng.integers(0,2,size=(8,8),dtype=np.uint8)*255
    m=GreenQuadWatermark(ProposedConfig(watermark_size=8))
    embed=m.embed(host,wm,key=b'right-key')
    meta=embed.metadata
    assert meta['fully_blind'] is True and meta['side_information_bits']==0
    assert meta['total_cells']==64 and meta['correction_failures']==0
    assert 0<=meta['fallback_cells']<=64
    bad=m.extract(embed.image,key=b'wrong-key',watermark_shape=wm.shape).watermark
    assert not np.array_equal(wm,bad) # sanity only; not a cryptographic guarantee
