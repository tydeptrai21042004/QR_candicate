import numpy as np
from qrwatermark.proposed.r_branch import build_r_candidate,r_llr,r12_value,qim_displacement_for_block


def test_r12_branch_is_sign_stable_and_encodes_both_bits():
    rng=np.random.default_rng(2)
    ok=0
    for _ in range(100):
        block=rng.uniform(20,235,size=(2,2))
        for bit in (0,1):
            cand=np.clip(np.rint(build_r_candidate(block,bit,48.0)),0,255)
            pred=int(r_llr(cand,48.0)>=0)
            ok += (pred==bit)
    assert ok >= 190


def test_r12_exact_continuous_distortion_identity():
    rng=np.random.default_rng(9)
    block=rng.uniform(30,220,size=(2,2))
    before=r12_value(block)
    delta,feasible=qim_displacement_for_block(block,1,48.0)
    assert feasible
    cand=build_r_candidate(block,1,48.0)
    measured=np.mean((cand-block)**2)
    expected=(delta*delta)/4.0
    assert abs(measured-expected) < 1e-8
