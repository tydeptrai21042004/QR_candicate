import numpy as np
from qrwatermark.baselines import Nha2022ImprovedQR,Su2017Hessenberg,Su2020Schur
from qrwatermark.evaluation.metrics import ber


def test_baselines_execute_clean_roundtrip():
    rng=np.random.default_rng(5)
    host=rng.integers(30,226,size=(64,64,3),dtype=np.uint8)
    wm=(rng.integers(0,2,size=(8,8),dtype=np.uint8)*255).astype(np.uint8)
    for method in (Nha2022ImprovedQR(),Su2017Hessenberg(),Su2020Schur()):
        emb=method.embed(host,wm,key=b'base')
        ext=method.extract(emb.image,key=b'base',side_info=emb.side_info,watermark_shape=wm.shape)
        assert ext.watermark.shape==wm.shape
        assert 0.0 <= ber(wm,ext.watermark) <= 1.0
