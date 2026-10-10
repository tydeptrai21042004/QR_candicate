"""Tests for the independent eight-pixel integer-convolution research carrier."""
from pathlib import Path

import numpy as np
import pytest

from qrwatermark.core.config import ProposedConfig
from qrwatermark.core.factory import build_method
from qrwatermark.evaluation.metrics import ber
from qrwatermark.proposed import ConvolutionCertifiedR12QIM
from qrwatermark.proposed.integer_convolution_v7 import _distribute_exact, _lattice_centres


def _method(n=8, period=100):
    return ConvolutionCertifiedR12QIM(ProposedConfig(design='integer_conv8_v7',
        watermark_size=n, channel=1, integer_conv_period=period, compute_certificate=False))


@pytest.mark.parametrize('seed', range(100))
def test_clean_roundtrip_random(seed):
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(64,64,3), dtype=np.uint8)
    wm = rng.integers(0,2,size=(8,8),dtype=np.uint8)*255
    model = _method()
    result = model.embed(img, wm, key=f'integer-qr-{seed}'.encode())
    decoded = model.extract(result.image, key=f'integer-qr-{seed}'.encode(), watermark_shape=(8,8))
    assert ber(wm, decoded.watermark)==0
    assert result.side_info is None
    assert result.metadata['fully_blind']
    assert np.array_equal(img[:,:,0],result.image[:,:,0])
    assert np.array_equal(img[:,:,2],result.image[:,:,2])


@pytest.mark.parametrize('value',[0,1,127,128,254,255])
@pytest.mark.parametrize('period',[4,52,100,124])
def test_constant_ends(value, period):
    x=np.full((64,64,3),value,dtype=np.uint8)
    wm=np.random.default_rng(52).integers(0,2,(8,8),dtype=np.uint8)*255
    method=_method(period=period)
    result=method.embed(x,wm,key=b'edge')
    assert ber(wm,method.extract(result.image,key=b'edge',watermark_shape=(8,8)).watermark)==0


def test_distribute_exact_saturated_fuzz():
    rng=np.random.default_rng(9)
    for _ in range(30):
        values=rng.choice(np.array([0,1,2,3,127,252,253,254,255],dtype=np.uint8),(4096,8))
        bits=rng.integers(0,2,size=4096,dtype=np.uint8)
        sums=values.astype(np.int16).sum(axis=1)
        targets=_lattice_centres(sums,bits,100)
        updated=_distribute_exact(values,targets)
        np.testing.assert_array_equal(updated.astype(np.int32).sum(axis=1), targets)
        np.testing.assert_array_equal((updated.astype(np.int32).sum(axis=1)%100)>=50, bits.astype(bool))


def test_wrong_key_and_invalid_period():
    rng=np.random.default_rng(56)
    host=rng.integers(30,220,size=(64,64,3),dtype=np.uint8)
    wm=rng.integers(0,2,(8,8),dtype=np.uint8)*255
    model=_method()
    img=model.embed(host,wm,key=b'correct').image
    assert ber(wm,model.extract(img,key=b'wrong',watermark_shape=(8,8)).watermark)>0.15
    for v in [0,3,7,-4,100.2]:
        with pytest.raises(ValueError,match='integer_conv_period'):
            ProposedConfig(design='integer_conv8_v7',integer_conv_period=v).validate()


def test_factory_and_mathematical_identity():
    p=Path(__file__).resolve().parents[1]/'configs/methods/proposed_integer_conv8_v7_experimental.yaml'
    m=build_method('blind_integer_convolution_v7_experimental',p)
    assert m.name=='blind_integer_convolution_v7_experimental'
    # QR of [q,b] with q = 1/sqrt(8) yields r12 = sum(b)/sqrt(8).
    rng=np.random.default_rng(8)
    b=rng.normal(size=8)
    q=np.ones(8)/np.sqrt(8)
    a=np.column_stack([q,b])
    _,r=np.linalg.qr(a,mode='reduced')
    # QR column signs are implementation-dependent. Fix the prescribed
    # first direction by multiplying the returned r12 by its Q sign.
    q_mat,_ = np.linalg.qr(a,mode="reduced")
    sign = np.sign(q_mat[:,0]@q)
    assert np.isclose(sign*r[0,1],b.sum()/np.sqrt(8),atol=1e-12)
