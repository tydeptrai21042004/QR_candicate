import numpy as np
from qrwatermark.core.config import ProposedConfig,NLMConfig
from qrwatermark.proposed import ProposedAdaptiveQR
from qrwatermark.evaluation.metrics import ber


def test_proposed_clean_roundtrip_small():
    rng=np.random.default_rng(3)
    host=rng.integers(20,236,size=(64,64,3),dtype=np.uint8)
    wm=(rng.integers(0,2,size=(8,8),dtype=np.uint8)*255).astype(np.uint8)
    cfg=ProposedConfig(watermark_size=8,repetition=3,nlm=NLMConfig(enabled=False))
    method=ProposedAdaptiveQR(cfg)
    emb=method.embed(host,wm,key=b'test')
    ext=method.extract(emb.image,key=b'test',side_info=emb.side_info,watermark_shape=wm.shape)
    assert ber(wm,ext.watermark) <= 0.02
    assert len(emb.side_info['packed_modes']) == 8  # 64 mode bits -> 8 bytes before npz compression
