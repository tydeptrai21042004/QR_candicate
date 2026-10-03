from __future__ import annotations

import numpy as np

from qrwatermark.core.config import ProposedConfig
from qrwatermark.evaluation.metrics import ber
from qrwatermark.proposed import ConvolutionCertifiedR12QIM
from qrwatermark.proposed.side_info import side_info_overhead_bits, unpack_selector_indices


def _cfg(**kw):
    base=dict(
        design="elegant_v2", watermark_size=8, qim_period=64.0,
        carrier_pool=4, carrier_selector_policy="min_energy",
        compute_certificate=False, store_certificate_mask=False,
    )
    base.update(kw)
    return ProposedConfig(**base)


def test_single_carrier_clean_roundtrip_and_selector_side_info():
    rng=np.random.default_rng(91)
    host=rng.integers(20,236,size=(64,64,3),dtype=np.uint8)
    wm=(rng.integers(0,2,size=(8,8),dtype=np.uint8)*255).astype(np.uint8)
    method=ConvolutionCertifiedR12QIM(_cfg())
    emb=method.embed(host,wm,key=b"single-carrier")
    ext=method.extract(emb.image,key=b"single-carrier",side_info=emb.side_info,watermark_shape=wm.shape)
    assert ber(wm,ext.watermark)==0.0
    selectors,_=unpack_selector_indices(emb.side_info,b"single-carrier")
    assert selectors.size==wm.size
    assert np.all(selectors<4)
    assert "packed_period_codes" not in emb.side_info
    assert emb.metadata["payload_embeddings_per_bit"]==1
    assert emb.metadata["repetition_used"] is False


def test_selector_overhead_is_two_bits_per_payload_for_pool4():
    rng=np.random.default_rng(92)
    host=rng.integers(30,226,size=(64,64,3),dtype=np.uint8)
    wm=(rng.integers(0,2,size=(8,8),dtype=np.uint8)*255).astype(np.uint8)
    method=ConvolutionCertifiedR12QIM(_cfg())
    emb=method.embed(host,wm,key=b"overhead")
    ov=side_info_overhead_bits(emb.side_info)
    assert ov["selector_code_bits"]==2*wm.size
    assert ov["period_code_bits"]==0


def test_no_certificate_does_not_change_embedding_pixels():
    rng=np.random.default_rng(93)
    host=rng.integers(25,231,size=(64,64,3),dtype=np.uint8)
    wm=(rng.integers(0,2,size=(8,8),dtype=np.uint8)*255).astype(np.uint8)
    a=ConvolutionCertifiedR12QIM(_cfg(compute_certificate=True,convolution_gaussian_sigmas=(0.5,0.75)))
    b=ConvolutionCertifiedR12QIM(_cfg(compute_certificate=False))
    ea=a.embed(host,wm,key=b"cert-independent")
    eb=b.embed(host,wm,key=b"cert-independent")
    assert np.array_equal(ea.image,eb.image)
