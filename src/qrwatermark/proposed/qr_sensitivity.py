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


def _bound_from_column_norms(base_block, first_column_perturbation, second_column_perturbation):
    base=np.asarray(base_block,dtype=np.float64)
    if base.shape!=(2,2): raise ValueError("implemented for 2x2 blocks")
    a,b=base[:,0],base[:,1]
    na=float(np.linalg.norm(a)); ne=float(first_column_perturbation); nf=float(second_column_perturbation)
    if na<=1e-15 or ne>=na:
        return float('inf')
    return float(nf + float(np.linalg.norm(b))*ne/(na-ne))


def finite_r12_perturbation_bound(base_block,perturbed_block):
    base=np.asarray(base_block,dtype=np.float64); pert=np.asarray(perturbed_block,dtype=np.float64)
    if base.shape!=(2,2) or pert.shape!=(2,2): raise ValueError("implemented for 2x2 blocks")
    a=base[:,0]; e=pert[:,0]-base[:,0]; f=pert[:,1]-base[:,1]
    na=float(np.linalg.norm(a)); ne=float(np.linalg.norm(e)); nf=float(np.linalg.norm(f))
    actual=abs(r12_value(pert)-r12_value(base))
    bound=_bound_from_column_norms(base,ne,nf)
    return R12PerturbationBound(float(actual),float(bound),ne,nf,na,r12_condition_number(base),bool(np.isfinite(bound)))


def group_pair_bound(base_blocks, perturbed_blocks, weights=None):
    """Deterministic bound for a single group perturbation.

    This is used for integer rounding and any other known base->perturbed pair.
    The returned theorem bound is a weighted sum of per-block finite R12 bounds.
    """
    if not base_blocks or len(base_blocks)!=len(perturbed_blocks):
        raise ValueError("base_blocks and perturbed_blocks must be non-empty and have equal length")
    w=unit_spread_weights(len(base_blocks)) if weights is None else np.asarray(weights,dtype=np.float64).ravel()
    w=w/np.linalg.norm(w)
    per=[finite_r12_perturbation_bound(a,b) for a,b in zip(base_blocks,perturbed_blocks)]
    bound=float(np.sum(np.abs(w)*np.asarray([x.certified_bound for x in per],dtype=np.float64)))
    base_stat=float(w@np.asarray([r12_value(b) for b in base_blocks]))
    pert_stat=float(w@np.asarray([r12_value(b) for b in perturbed_blocks]))
    return bound,abs(pert_stat-base_stat),tuple(float(x.certified_bound) for x in per)


def convex_hull_group_bound(base_blocks,extreme_block_sets,weights=None):
    """Certify the spread statistic for every convex mixture of extreme images.

    If an attacked block is A + sum_l alpha_l E_l, then the perturbation-column
    norms satisfy ||e|| <= max_l ||e_l|| and ||f|| <= max_l ||f_l||.  Applying
    the finite R12 inequality with these two maxima gives a conservative bound
    that is valid for *all* convex mixtures, even when different extremes
    maximize the two columns.  This is stronger/correcter than taking the max
    of already-combined per-extreme nonlinear bounds.
    """
    if not base_blocks: raise ValueError("base_blocks must be non-empty")
    w=unit_spread_weights(len(base_blocks)) if weights is None else np.asarray(weights,dtype=np.float64).ravel()
    w=w/np.linalg.norm(w)
    if not extreme_block_sets:
        return 0.0,0.0,[]
    for blocks in extreme_block_sets:
        if len(blocks)!=len(base_blocks): raise ValueError("each extreme set must match base_blocks")

    block_bounds=[]
    for i,base in enumerate(base_blocks):
        ne=[]; nf=[]
        for blocks in extreme_block_sets:
            e=np.asarray(blocks[i],dtype=np.float64)-np.asarray(base,dtype=np.float64)
            ne.append(float(np.linalg.norm(e[:,0])))
            nf.append(float(np.linalg.norm(e[:,1])))
        block_bounds.append(_bound_from_column_norms(base,max(ne),max(nf)))
    theorem=float(np.sum(np.abs(w)*np.asarray(block_bounds,dtype=np.float64)))

    base_stat=float(w@np.asarray([r12_value(b) for b in base_blocks]))
    observed=[]; extreme_pair_bounds=[]
    for blocks in extreme_block_sets:
        stat=float(w@np.asarray([r12_value(b) for b in blocks]))
        observed.append(abs(stat-base_stat))
        bnd,_,_=group_pair_bound(base_blocks,blocks,w)
        extreme_pair_bounds.append(float(bnd))
    return theorem,float(max(observed)),extreme_pair_bounds
