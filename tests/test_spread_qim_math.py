import numpy as np
from qrwatermark.proposed.convolution import gaussian_kernel,reflect_convolve
from qrwatermark.proposed.convolution_uncertainty import convex_mixture
from qrwatermark.proposed.qr_sensitivity import convex_hull_group_bound,finite_r12_perturbation_bound
from qrwatermark.proposed.r_branch import r12_value
from qrwatermark.proposed.spread_qim import minimum_energy_box_projection,spread_embedding_for_blocks,unit_spread_weights

def test_minimum_energy_projection_closed_form():
    w=unit_spread_weights(5); d=7.5; delta,ok=minimum_energy_box_projection(d,w,np.full(5,-100.0),np.full(5,100.0)); assert ok; assert np.allclose(delta,d*w); assert np.isclose(w@delta,d)

def test_spread_embedding_feasible_and_finite():
    rng=np.random.default_rng(4); blocks=[rng.uniform(30,220,size=(2,2)) for _ in range(5)]; emb=spread_embedding_for_blocks(blocks,1,32.0); assert emb.feasible and np.isfinite(emb.energy)

def test_finite_r12_bound_dominates_actual():
    a=np.array([[80.,120.],[100.,140.]]); b=a+np.array([[1.,-2.],[-1.5,1.]])
    r=finite_r12_perturbation_bound(a,b); assert r.finite and r.actual_shift<=r.certified_bound+1e-12

def test_convex_hull_extreme_bound_covers_unseen_mixture():
    rng=np.random.default_rng(12); image=rng.uniform(40,210,size=(32,32)); k1=gaussian_kernel(3,.5); k2=gaussian_kernel(3,.75); y1=reflect_convolve(image,k1); y2=reflect_convolve(image,k2); ym=reflect_convolve(image,convex_mixture([k1,k2],np.array([.37,.63])))
    pos=[(8,8),(8,12),(12,8),(12,12),(16,16)]; base=[image[r:r+2,c:c+2] for r,c in pos]; e1=[y1[r:r+2,c:c+2] for r,c in pos]; e2=[y2[r:r+2,c:c+2] for r,c in pos]; mix=[ym[r:r+2,c:c+2] for r,c in pos]
    bound,_,_=convex_hull_group_bound(base,[e1,e2]); w=unit_spread_weights(len(pos)); actual=abs(float(w@np.array([r12_value(x) for x in mix]))-float(w@np.array([r12_value(x) for x in base]))); assert actual<=bound+1e-10

from qrwatermark.proposed.robust_allocation import solve_minimax_allocation

def test_minimax_allocation_balances_unknown_convolution_scenarios():
    # Carrier 1 is best for attack 1, carrier 2 for attack 2; robust optimum mixes them.
    S=np.array([[1.0,0.2],[0.2,1.0]])
    result=solve_minimax_allocation(S)
    assert result.success
    assert np.allclose(result.weights,[0.5,0.5],atol=1e-8)
    assert np.isclose(result.guaranteed_margin,0.6,atol=1e-8)
