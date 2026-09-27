from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..core.config import ProposedConfig
from .qr_sensitivity import convex_hull_group_bound
from .r_branch import qim_displacement,qim_displacement_for_block
from .spread_qim import spread_embedding_for_blocks,unit_spread_weights

@dataclass(frozen=True)
class PeriodDecision:
    period_index:int; period:float; convolution_shift:float; required_margin:float; certificate_margin:float; worst_case_mse:float; certified:bool; distortion_feasible:bool; range_feasible:bool
    theorem_bound:float=0.0
    extreme_bounds:tuple[float,...]=()

def theoretical_group_mse(r12_values,period,blocks=None):
    values=np.asarray(r12_values,dtype=np.float64).ravel()
    if blocks is not None and len(blocks)!=values.size: raise ValueError("blocks and r12_values must have same length")
    bit_mse=[]; all_ok=True
    for bit in (0,1):
        ds=[]
        for i,v in enumerate(values):
            if blocks is None: d=qim_displacement(float(v),bit,float(period)); ok=True
            else: d,ok=qim_displacement_for_block(blocks[i],bit,float(period))
            ds.append(d); all_ok=all_ok and ok
        d=np.asarray(ds); bit_mse.append(float(np.mean((d*d)/4.0)))
    return max(bit_mse),bool(all_ok)

def theoretical_spread_group_mse(blocks,period):
    w=unit_spread_weights(len(blocks)); vals=[]; ok=True
    for bit in (0,1):
        emb=spread_embedding_for_blocks(blocks,bit,float(period),w); ok=ok and emb.feasible; vals.append(float(emb.energy/(4.0*len(blocks))))
    return max(vals),bool(ok)

def choose_spread_period_for_group(blocks,convolved_block_sets,cfg:ProposedConfig):
    w=unit_spread_weights(len(blocks)); theorem,observed,extreme=convex_hull_group_bound(blocks,convolved_block_sets,w)
    required=float(cfg.convolution_safety_factor)*theorem+float(cfg.additive_feature_budget)+float(cfg.rounding_feature_budget)
    ds=[]
    for idx,p0 in enumerate(cfg.period_candidates):
        p=float(p0); mse,range_ok=theoretical_spread_group_mse(blocks,p); residual=p/4.0-required; dist_ok=mse<=float(cfg.max_group_mse)
        ds.append(PeriodDecision(idx,p,observed,required,residual,mse,bool(np.isfinite(theorem) and residual>=0),dist_ok,range_ok,theorem,tuple(float(x) for x in extreme)))
    feasible=[d for d in ds if d.certified and d.distortion_feasible and d.range_feasible]
    if feasible: return max(feasible,key=lambda d:(d.certificate_margin,-d.worst_case_mse,d.period))
    feasible=[d for d in ds if d.distortion_feasible and d.range_feasible]
    if feasible: return max(feasible,key=lambda d:(d.certificate_margin,-d.worst_case_mse,d.period))
    return min(ds,key=lambda d:(not d.range_feasible,d.worst_case_mse,d.period))

def choose_period_for_group(r12_values,convolved_r12_values,cfg:ProposedConfig,blocks=None):
    base=np.asarray(r12_values,dtype=np.float64).ravel(); conv=np.asarray(convolved_r12_values,dtype=np.float64)
    if conv.ndim!=2 or conv.shape[1]!=base.size: raise ValueError("convolved_r12_values must have shape (n_kernels, repetition)")
    shift=0.0 if conv.shape[0]==0 else float(np.max(np.abs(conv-base[None,:])))
    required=float(cfg.convolution_safety_factor)*shift+float(cfg.additive_feature_budget)+float(cfg.rounding_feature_budget)
    ds=[]
    for idx,p0 in enumerate(cfg.period_candidates):
        p=float(p0); mse,range_ok=theoretical_group_mse(base,p,blocks); residual=p/4.0-required
        ds.append(PeriodDecision(idx,p,shift,required,residual,mse,residual>=0,mse<=float(cfg.max_group_mse),range_ok))
    feasible=[d for d in ds if d.distortion_feasible and d.range_feasible]
    return max(feasible or ds,key=lambda d:(d.certificate_margin,-d.worst_case_mse,d.period))
