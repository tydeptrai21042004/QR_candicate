import numpy as np
from qrwatermark.proposed.q_branch import build_q_candidate,q_llr


def test_q_branch_encodes_both_bits_after_rounding():
    rng=np.random.default_rng(1)
    ok=0
    for _ in range(100):
        block=rng.uniform(20,235,size=(2,2))
        for bit in (0,1):
            cand=np.clip(np.rint(build_q_candidate(block,bit,0.12)),0,255)
            pred=int(q_llr(cand,0.12)>=0)
            ok += (pred==bit)
    assert ok >= 190
