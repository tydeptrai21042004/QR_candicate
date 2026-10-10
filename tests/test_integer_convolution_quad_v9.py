import numpy as np
import pytest
from qrwatermark.core.config import ProposedConfig
from qrwatermark.proposed import ConvolutionCertifiedR12QIM
from qrwatermark.proposed.integer_convolution_quad_v9 import _to_cells, _to_patches
from qrwatermark.evaluation.metrics import ber

@pytest.mark.parametrize('seed',range(100))
def test_clean(seed):
    rng=np.random.default_rng(seed)
    host=rng.integers(0,256,(64,64,3),dtype=np.uint8)
    wm=rng.integers(0,2,(8,8),dtype=np.uint8)*255
    m=ConvolutionCertifiedR12QIM(ProposedConfig(design='integer_conv_quad_v9',watermark_size=8,channel=1))
    key=f'quad-{seed}'.encode()
    stego=m.embed(host,wm,key=key)
    assert stego.side_info is None
    assert ber(wm,m.extract(stego.image,key=key,watermark_shape=(8,8)).watermark)==0

@pytest.mark.parametrize('value',[0,1,127,128,254,255])
def test_constant(value):
    m=ConvolutionCertifiedR12QIM(ProposedConfig(design='integer_conv_quad_v9',watermark_size=8,channel=1))
    x=np.full((64,64,3),value,dtype=np.uint8)
    wm=np.random.default_rng(44).integers(0,2,(8,8),dtype=np.uint8)*255
    y=m.embed(x,wm,key=b'quad-constant').image
    assert ber(wm,m.extract(y,key=b'quad-constant',watermark_shape=(8,8)).watermark)==0

def test_layout_inverse():
    p=np.arange(3*16,dtype=np.uint8).reshape(-1,4,4)
    np.testing.assert_array_equal(_to_patches(_to_cells(p)),p)
