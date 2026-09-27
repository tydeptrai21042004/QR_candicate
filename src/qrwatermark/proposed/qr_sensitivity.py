from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .r_branch import r12_value
from .spread_qim import unit_spread_weights

@dataclass(frozen=True)
class R12PerturbationBound:
    actual_shift: float
    certified_bound: float
    first_column_perturbation: float
    second_column_perturbation: float
    first_column_norm: float
    condition_number: float
    finite: bool

def r12_condition_number(block):
    a=np.asarray(block,dtype=np.float64); x=a[:,0]; y=a[:,1]; n=float(np.linalg.norm(x))
    if n<=1e-15: return float('inf')
    q=x/n; perp=y-q*float(q@y)
    return float(np.sqrt(1.0+float(perp@perp)/(n*n)))

def finite_r12_perturbation_bound(base_block,perturbed_block):
    base=np.asarray(base_block,dtype=np.float64); pert=np.asarray(perturbed_block,dtype=np.float64)
    if base.shape!=(2,2) or pert.shape!=(2,2): raise ValueError("implemented for 2x2 blocks")
    a,b=base[:,0],base[:,1]; e,f=pert[:,0]-a,pert[:,1]-b
    na=float(np.linalg.norm(a)); ne=float(np.linalg.norm(e)); nf=float(np.linalg.norm(f)); actual=abs(r12_value(pert)-r12_value(base))
    if na<=1e-15 or ne>=na: bound=float('inf'); finite=False
    else: bound=nf+float(np.linalg.norm(b))*ne/(na-ne); finite=bool(np.isfinite(bound))
    return R12PerturbationBound(float(actual),float(bound),ne,nf,na,r12_condition_number(base),finite)

def convex_hull_group_bound(base_blocks,extreme_block_sets,weights=None):
    if not base_blocks: raise ValueError("base_blocks must be non-empty")
    w=unit_spread_weights(len(base_blocks)) if weights is None else np.asarray(weights,dtype=np.float64).ravel(); w=w/np.linalg.norm(w)
    base_stat=float(w@np.asarray([r12_value(b) for b in base_blocks]))
    bounds=[]; observed=[]
    for blocks in extreme_block_sets:
        if len(blocks)!=len(base_blocks): raise ValueError("each extreme set must match base_blocks")
        cb=[finite_r12_perturbation_bound(a,b).certified_bound for a,b in zip(base_blocks,blocks)]
        bounds.append(float(np.sum(np.abs(w)*np.asarray(cb))))
        observed.append(abs(float(w@np.asarray([r12_value(b) for b in blocks]))-base_stat))
    if not bounds: return 0.0,0.0,[]
    return float(np.max(bounds)),float(np.max(observed)),bounds
