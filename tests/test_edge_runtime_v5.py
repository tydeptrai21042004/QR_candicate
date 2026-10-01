from __future__ import annotations

import hashlib
import hmac
from pathlib import Path

import numpy as np

from qrwatermark.core.factory import build_method
from qrwatermark.proposed.certificate import certify_spread_group, certify_spread_groups_arrays
from qrwatermark.proposed.soft_decoder import _decode_groups_reference, decode_groups
from qrwatermark.utils.image_io import read_color
from qrwatermark.utils.permutation import clear_permutation_cache, keyed_permutation, selected_block_positions
from qrwatermark.utils.watermark import prepare_binary_watermark


def _reference_permutation(length: int, key: bytes, domain: bytes = b"qrwatermark:block-order:v2"):
    rec=[]
    for i in range(length):
        rec.append((hmac.new(key,domain+i.to_bytes(8,"big"),hashlib.sha256).digest(),i))
    rec.sort(key=lambda x:x[0])
    return np.asarray([i for _,i in rec],dtype=np.int64)


def test_cached_permutation_is_bit_exact():
    clear_permutation_cache()
    key=b"edge-cache-test"
    ref=_reference_permutation(512,key)
    a=keyed_permutation(512,key)
    b=keyed_permutation(512,key)
    assert np.array_equal(a,ref)
    assert np.array_equal(b,ref)
    # Returned arrays must not alias the cache.
    a[0]=-1
    c=keyed_permutation(512,key)
    assert np.array_equal(c,ref)


def test_vectorized_decoder_matches_scalar_reference():
    rng=np.random.default_rng(20261001)
    channel=rng.integers(0,256,size=(96,96),dtype=np.uint8).astype(np.float64)
    repetition=5; groups=128; periods=np.asarray([24.,28.,32.,36.])
    p=periods[rng.integers(0,4,size=groups)]
    positions=selected_block_positions(channel.shape,2,groups*repetition,b"edge-decode-test")
    rb,rc,rv=_decode_groups_reference(channel,positions,p,repetition,2)
    vb,vc,vv=decode_groups(channel,positions,p,repetition,2)
    assert np.array_equal(vb,rb)
    assert np.allclose(vv,rv,rtol=0.0,atol=3e-14)
    assert abs(vc-rc)<=3e-14


def test_batch_certificate_matches_scalar_generic_and_tight_path():
    rng=np.random.default_rng(17)
    groups=12; rep=5
    rounded=rng.uniform(25,230,size=(groups,rep,2,2))
    continuous=rounded+rng.uniform(-0.45,0.45,size=rounded.shape)
    e0=rounded+rng.normal(0,0.18,size=rounded.shape)
    e1=rounded+rng.normal(0,0.24,size=rounded.shape)
    ext=np.stack([e0,e1],axis=0)
    period_idx=rng.integers(0,4,size=groups,dtype=np.uint8)
    method=build_method("proposed","configs/methods/proposed.yaml")
    cfg=method.config
    for tight in (False,True):
        st=certify_spread_groups_arrays(continuous,rounded,ext,period_idx,cfg,tighten_final=tight)
        for k in range(groups):
            scalar=certify_spread_group(
                list(continuous[k]),list(rounded[k]),[list(e0[k]),list(e1[k])],0,int(period_idx[k]),cfg,tighten_final=tight
            )
            assert bool(st['certified'][k])==scalar.certified
            assert np.isclose(st['convolution_bound'][k],scalar.theorem_bound,rtol=1e-12,atol=1e-12)
            assert np.isclose(st['rounding_bound'][k],scalar.rounding_bound,rtol=1e-12,atol=1e-12)
            assert np.isclose(st['certificate_margin'][k],scalar.certificate_margin,rtol=1e-12,atol=1e-12)


def test_edge_fast_path_preserves_known_v4_output_hash():
    root=Path(__file__).resolve().parents[1]
    method=build_method("proposed",root/"configs/methods/proposed.yaml")
    method.config.watermark_size=16
    host=read_color(root/"data/hosts/classical/girl.bmp")
    wm=prepare_binary_watermark(root/"data/watermarks/watermark_1.png",16)
    res=method.embed(host,wm,key=b"edge-eq-2026")
    digest=hashlib.sha256(res.image.tobytes()).hexdigest()
    assert digest=="ca2d672a72e9a5c9a27867bbdde1d4e49dd819160dd451f2dc3cbee29ff0d5b7"
    assert res.metadata['certified_fraction']==0.8203125
