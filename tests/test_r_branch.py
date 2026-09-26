import numpy as np
from qrwatermark.proposed.r_branch import build_r_candidate,r_llr


def test_r12_branch_is_sign_stable_and_encodes_both_bits():
    rng=np.random.default_rng(2)
    ok=0
    for _ in range(100):
        block=rng.uniform(20,235,size=(2,2))
        for bit in (0,1):
            cand=np.clip(np.rint(build_r_candidate(block,bit,8.0)),0,255)
            pred=int(r_llr(cand,8.0)>=0)
            ok += (pred==bit)
    assert ok >= 190
