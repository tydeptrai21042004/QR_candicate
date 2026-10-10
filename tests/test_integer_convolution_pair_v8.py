"""Tests for two independent low-pass carriers in each keyed 4x4 block."""
import numpy as np
import pytest
from qrwatermark.core.config import ProposedConfig
from qrwatermark.proposed import ConvolutionCertifiedR12QIM
from qrwatermark.proposed.integer_convolution_pair_v8 import _to_carriers,_to_patches
from qrwatermark.evaluation.metrics import ber


def method():
    return ConvolutionCertifiedR12QIM(ProposedConfig(design='integer_conv_pair_v8',watermark_size=8,channel=1))


@pytest.mark.parametrize('seed',range(100))
def test_random_clean_roundtrip(seed):
    rng=np.random.default_rng(seed)
    host=rng.integers(0,256,(64,64,3),dtype=np.uint8)
    wm=rng.integers(0,2,(8,8),dtype=np.uint8)*255
    m=method();key=f'pair-{seed}'.encode()
    e=m.embed(host,wm,key=key)
    assert e.side_info is None
    assert e.metadata['fully_blind']
    extracted=m.extract(e.image,key=key,watermark_shape=(8,8))
    assert ber(wm,extracted.watermark)==0
    assert np.array_equal(host[:,:,0],e.image[:,:,0])
    assert np.array_equal(host[:,:,2],e.image[:,:,2])


@pytest.mark.parametrize('constant',[0,1,127,128,254,255])
def test_constant(constant):
    host=np.full((64,64,3),constant,dtype=np.uint8)
    wm=np.random.default_rng(777).integers(0,2,(8,8),dtype=np.uint8)*255
    m=method(); key=b'pair-constant'
    e=m.embed(host,wm,key=key)
    assert ber(wm,m.extract(e.image,key=key,watermark_shape=(8,8)).watermark)==0


def test_carrier_partition_is_lossless():
    p=np.arange(2*4*4,dtype=np.uint8).reshape(-1,4,4)
    result=_to_carriers(p)
    assert result.shape==(2,2,8)
    np.testing.assert_array_equal(result,_to_carriers(_to_patches(result)))
    np.testing.assert_array_equal(p,_to_patches(result))
