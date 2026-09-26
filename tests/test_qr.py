import numpy as np
from qrwatermark.proposed.qr import canonical_qr


def test_canonical_qr_reconstructs_and_positive_diagonal():
    rng=np.random.default_rng(0)
    for _ in range(100):
        a=rng.uniform(0,255,size=(2,2))
        q,r=canonical_qr(a)
        assert np.allclose(q@r,a,atol=1e-9)
        assert np.all(np.diag(r)>=-1e-12)
        assert np.allclose(q.T@q,np.eye(2),atol=1e-9)
