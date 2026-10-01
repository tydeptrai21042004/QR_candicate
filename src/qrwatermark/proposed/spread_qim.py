from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .r_branch import apply_r12_delta, qim_phase_decision, qim_target, r12_displacement_interval, r12_value

@dataclass(frozen=True)
class SpreadEmbedding:
    deltas: np.ndarray
    statistic: float
    target: float
    displacement: float
    energy: float
    feasible: bool

def unit_spread_weights(count: int) -> np.ndarray:
    if int(count) < 1:
        raise ValueError("count must be positive")
    return np.full(int(count), 1.0 / np.sqrt(float(count)), dtype=np.float64)

def group_statistic(values: np.ndarray, weights: np.ndarray | None = None) -> float:
    values=np.asarray(values,dtype=np.float64).ravel()
    w=unit_spread_weights(values.size) if weights is None else np.asarray(weights,dtype=np.float64).ravel()
    if values.size != w.size: raise ValueError("values and weights must have the same length")
    n=float(np.linalg.norm(w))
    if n<=0: raise ValueError("weights must have non-zero norm")
    w=w/n
    return float(w@values)

def _weighted_box_range(w,lo,hi):
    low=np.where(w>=0,w*lo,w*hi); high=np.where(w>=0,w*hi,w*lo)
    return float(low.sum()),float(high.sum())

def minimum_energy_box_projection(displacement,weights,lower,upper,*,tol=1e-11,max_iter=120):
    w=np.asarray(weights,dtype=np.float64).ravel(); lo=np.asarray(lower,dtype=np.float64).ravel(); hi=np.asarray(upper,dtype=np.float64).ravel()
    if not (w.size==lo.size==hi.size): raise ValueError("weights/lower/upper must have equal length")
    if np.any(lo>hi): return np.zeros_like(w),False
    n=float(np.linalg.norm(w))
    if n<=0: raise ValueError("weights must have non-zero norm")
    w=w/n; d=float(displacement)
    mn,mx=_weighted_box_range(w,lo,hi)
    if d<mn-tol or d>mx+tol: return np.zeros_like(w),False
    unconstrained=d*w
    if np.all(unconstrained>=lo-tol) and np.all(unconstrained<=hi+tol): return unconstrained.astype(np.float64),True
    def f(lam): return float(w@np.clip(lam*w,lo,hi))
    left,right=-1.0,1.0
    while f(left)>d and abs(left)<1e18: left*=2.0
    while f(right)<d and abs(right)<1e18: right*=2.0
    for _ in range(max_iter):
        mid=.5*(left+right)
        if f(mid)<d: left=mid
        else: right=mid
        if right-left<=tol*max(1.0,abs(left),abs(right)): break
    delta=np.clip(.5*(left+right)*w,lo,hi)
    return delta.astype(np.float64),abs(float(w@delta-d))<=1e-6

def spread_embedding_for_blocks(blocks,bit,period,weights=None,*,pixel_min=0.0,pixel_max=255.0):
    if not blocks: raise ValueError("blocks must be non-empty")
    w=unit_spread_weights(len(blocks)) if weights is None else np.asarray(weights,dtype=np.float64).ravel()
    if w.size!=len(blocks): raise ValueError("weights and blocks must have the same length")
    w=w/np.linalg.norm(w)
    values=np.asarray([r12_value(b) for b in blocks],dtype=np.float64)
    stat=float(w@values); target=qim_target(stat,int(bit),float(period)); d=float(target-stat)
    ints=[r12_displacement_interval(b,pixel_min,pixel_max) for b in blocks]
    lo=np.asarray([x[0] for x in ints]); hi=np.asarray([x[1] for x in ints])
    delta,ok=minimum_energy_box_projection(d,w,lo,hi)
    return SpreadEmbedding(delta,stat,float(target),d,float(delta@delta),bool(ok))

def apply_spread_embedding(blocks,bit,period,weights=None):
    emb=spread_embedding_for_blocks(blocks,bit,period,weights)
    if not emb.feasible: return [np.asarray(b,dtype=np.float64).copy() for b in blocks],emb
    return [apply_r12_delta(block,float(delta)) for block,delta in zip(blocks,emb.deltas)],emb

def spread_decision(values,period,weights=None):
    """Return hardware-friendly ``(bit, signed_score, confidence)`` for a group."""
    stat=group_statistic(values,weights)
    return qim_phase_decision(stat,float(period))

def spread_llr(values,period,weights=None):
    """Backward-compatible name for the signed triangular QIM score."""
    _,score,_=spread_decision(values,period,weights)
    return float(score)
