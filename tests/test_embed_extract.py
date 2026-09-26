import numpy as np
from qrwatermark.core.config import ProposedConfig,NLMConfig
from qrwatermark.proposed import ConvolutionCertifiedR12QIM
from qrwatermark.evaluation.metrics import ber


def test_proposed_clean_roundtrip_small():
    rng=np.random.default_rng(3)
    host=rng.integers(20,236,size=(64,64,3),dtype=np.uint8)
    wm=(rng.integers(0,2,size=(8,8),dtype=np.uint8)*255).astype(np.uint8)
    cfg=ProposedConfig(
        watermark_size=8,
        repetition=3,
        period_candidates=(32.0,40.0,48.0),
        convolution_gaussian_sigmas=(0.5,),
        additive_feature_budget=1.0,
        rounding_feature_budget=0.5,
        nlm=NLMConfig(enabled=False),
    )
    method=ConvolutionCertifiedR12QIM(cfg)
    emb=method.embed(host,wm,key=b'test')
    ext=method.extract(emb.image,key=b'test',side_info=emb.side_info,watermark_shape=wm.shape)
    assert ber(wm,ext.watermark) <= 0.02
    assert emb.metadata['method']=='ccqr_r12_qim_v1'
    assert 0.0 <= emb.metadata['certified_fraction'] <= 1.0
    # 64 payload bits * 2 bits/code = 128 bits = 16 packed bytes.
    assert len(emb.side_info['packed_period_codes']) == 16
