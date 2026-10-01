from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from ..core.config import ProposedConfig
from .qr_sensitivity import convex_hull_group_bound,group_pair_bound,tightened_two_extreme_convex_group_bound
from .r_branch import qim_displacement,qim_displacement_for_block
from .spread_qim import spread_embedding_for_blocks,unit_spread_weights


@dataclass(frozen=True)
class PeriodDecision:
    period_index:int
    period:float
    convolution_shift:float
    required_margin:float
    certificate_margin:float
    worst_case_mse:float
    certified:bool
    distortion_feasible:bool
    range_feasible:bool
    theorem_bound:float=0.0
    extreme_bounds:tuple[float,...]=()
    rounding_bound:float=0.0
    rounding_shift:float=0.0
    status:str="uncertified"
    bound_mode:str="generic"
    fallback_bound:float=float("inf")
    path_interval_bounds:tuple[float,...]=()


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


def actual_spread_group_mse(blocks,bit,period):
    emb=spread_embedding_for_blocks(blocks,int(bit),float(period),unit_spread_weights(len(blocks)))
    return float(emb.energy/(4.0*len(blocks))),bool(emb.feasible)


def choose_initial_spread_period_for_group(blocks,bit,cfg:ProposedConfig):
    """Payload-independent initial period: minimize worst-bit continuous MSE.

    `bit` is accepted for API compatibility but deliberately not used.  This
    avoids the old largest-period bias without making the initial period code a
    direct function of the payload bit.
    """
    del bit
    choices=[]
    for idx,p0 in enumerate(cfg.period_candidates):
        p=float(p0); mse,range_ok=theoretical_spread_group_mse(blocks,p)
        dist_ok=bool(mse<=float(cfg.max_group_mse))
        choices.append(PeriodDecision(idx,p,0.0,0.0,p/4.0,mse,False,dist_ok,range_ok,status="pending_post_embedding_certificate"))
    feasible=[d for d in choices if d.distortion_feasible and d.range_feasible]
    if feasible:
        return min(feasible,key=lambda d:(d.worst_case_mse,d.period))
    return min(choices,key=lambda d:(not d.range_feasible,d.worst_case_mse,d.period))


def certify_spread_group(continuous_blocks,rounded_blocks,convolved_block_sets,bit,period_index,cfg:ProposedConfig,*,tighten_final=False):
    """Post-embedding certificate for the exact rounded watermarked image.

    The continuous spread embedding lands exactly on the intended QIM lattice.
    Integer rounding is bounded deterministically from continuous->rounded blocks;
    convolution is then bounded from rounded watermark->attacked watermark over
    the whole configured convex hull.  Triangle inequality gives a valid total
    displacement from the intended lattice point.
    """
    p=float(cfg.period_candidates[int(period_index)])
    w=unit_spread_weights(len(rounded_blocks))
    fallback=float("inf"); mode="generic"; path_intervals=()
    if tighten_final and len(convolved_block_sets)==2:
        conv_bound,observed_conv,extreme,fallback,mode,path_intervals=tightened_two_extreme_convex_group_bound(
            rounded_blocks,convolved_block_sets,w,subdivisions=int(cfg.certificate_path_subdivisions)
        )
    else:
        conv_bound,observed_conv,extreme=convex_hull_group_bound(rounded_blocks,convolved_block_sets,w)
    round_bound,observed_round,_=group_pair_bound(continuous_blocks,rounded_blocks,w)
    required=float(cfg.convolution_safety_factor)*conv_bound + round_bound + float(cfg.additive_feature_budget)
    margin=p/4.0-required
    mse=float(sum(np.sum((np.asarray(c)-np.asarray(r))**2) for c,r in zip(continuous_blocks,rounded_blocks))/(4.0*len(rounded_blocks)))
    finite=bool(np.isfinite(conv_bound) and np.isfinite(round_bound))
    certified=bool(finite and margin>=0.0)
    return PeriodDecision(
        int(period_index),p,float(observed_conv),float(required),float(margin),mse,
        certified,True,True,float(conv_bound),tuple(float(x) for x in extreme),
        float(round_bound),float(observed_round),"certified" if certified else "uncertified",
        str(mode),float(fallback),tuple(float(x) for x in path_intervals)
    )


# Backward-compatible helper retained for notebooks.  It is now explicitly a
# pre-embedding diagnostic and MUST NOT be interpreted as the final certificate.
def choose_spread_period_for_group(blocks,convolved_block_sets,cfg:ProposedConfig):
    w=unit_spread_weights(len(blocks)); theorem,observed,extreme=convex_hull_group_bound(blocks,convolved_block_sets,w)
    required=float(cfg.convolution_safety_factor)*theorem+float(cfg.additive_feature_budget)
    ds=[]
    for idx,p0 in enumerate(cfg.period_candidates):
        p=float(p0); mse,range_ok=theoretical_spread_group_mse(blocks,p); residual=p/4.0-required; dist_ok=mse<=float(cfg.max_group_mse)
        ds.append(PeriodDecision(idx,p,observed,required,residual,mse,bool(np.isfinite(theorem) and residual>=0),dist_ok,range_ok,theorem,tuple(float(x) for x in extreme),status="pre_embedding_diagnostic"))
    feasible=[d for d in ds if d.distortion_feasible and d.range_feasible]
    return min(feasible or ds,key=lambda d:(d.worst_case_mse,d.period))


def choose_period_for_group(r12_values,convolved_r12_values,cfg:ProposedConfig,blocks=None):
    base=np.asarray(r12_values,dtype=np.float64).ravel(); conv=np.asarray(convolved_r12_values,dtype=np.float64)
    if conv.ndim!=2 or conv.shape[1]!=base.size: raise ValueError("convolved_r12_values must have shape (n_kernels, repetition)")
    shift=0.0 if conv.shape[0]==0 else float(np.max(np.abs(conv-base[None,:])))
    required=float(cfg.convolution_safety_factor)*shift+float(cfg.additive_feature_budget)
    ds=[]
    for idx,p0 in enumerate(cfg.period_candidates):
        p=float(p0); mse,range_ok=theoretical_group_mse(base,p,blocks); residual=p/4.0-required
        ds.append(PeriodDecision(idx,p,shift,required,residual,mse,residual>=0,mse<=float(cfg.max_group_mse),range_ok,status="legacy_diagnostic"))
    feasible=[d for d in ds if d.distortion_feasible and d.range_feasible]
    return min(feasible or ds,key=lambda d:(d.worst_case_mse,d.period))



def certify_spread_groups_arrays(continuous_groups,rounded_groups,extreme_group_sets,period_indices,cfg:ProposedConfig,*,tighten_final=False,embedding_feasible=None):
    """Vectorized certificate state for the edge frame loop."""
    from .qr_sensitivity import (
        group_pair_bound_batch,convex_hull_group_bound_batch,
        tightened_two_extreme_convex_group_bound_batch,
    )
    cont=np.asarray(continuous_groups,dtype=np.float64)
    rnd=np.asarray(rounded_groups,dtype=np.float64)
    ext=np.asarray(extreme_group_sets,dtype=np.float64)
    idx=np.asarray(period_indices,dtype=np.int64).ravel(); n=idx.size
    if cont.shape!=rnd.shape or cont.shape[:1]!=(n,) or cont.shape[-2:]!=(2,2):
        raise ValueError("group arrays must have shape (groups,repetition,2,2)")
    if ext.ndim!=5 or ext.shape[1:]!=rnd.shape:
        raise ValueError("extreme_group_sets has incompatible shape")
    rep=rnd.shape[1]; w=unit_spread_weights(rep)
    fallback=np.full(n,np.inf,dtype=np.float64); use_path=np.zeros(n,dtype=bool); intervals=np.empty((0,n),dtype=np.float64)
    if tighten_final and ext.shape[0]==2:
        conv,obs,extreme,fallback,use_path,intervals=tightened_two_extreme_convex_group_bound_batch(
            rnd,ext,w,subdivisions=int(cfg.certificate_path_subdivisions)
        )
    else:
        conv,obs,extreme=convex_hull_group_bound_batch(rnd,ext,w)
    rb,robs,_=group_pair_bound_batch(cont,rnd,w)
    periods=np.asarray(cfg.period_candidates,dtype=np.float64)[idx]
    required=float(cfg.convolution_safety_factor)*conv+rb+float(cfg.additive_feature_budget)
    margin=periods/4.0-required
    mse=np.sum((cont-rnd)**2,axis=(1,2,3))/(4.0*float(rep))
    finite=np.isfinite(conv)&np.isfinite(rb)
    ef=np.ones(n,dtype=bool) if embedding_feasible is None else np.asarray(embedding_feasible,dtype=bool).ravel()
    if ef.size!=n: raise ValueError("embedding_feasible length mismatch")
    cert=finite&(margin>=0.0)&ef
    return {
        'periods':periods,'period_indices':idx,'convolution_bound':conv,
        'convolution_shift':obs,'extreme_bounds':extreme,'fallback_bound':fallback,
        'use_path':use_path,'path_interval_bounds':intervals,
        'rounding_bound':rb,'rounding_shift':robs,'required_margin':required,
        'certificate_margin':margin,'worst_case_mse':mse,'certified':cert,
        'embedding_feasible':ef,
    }


def certify_spread_groups_batch(continuous_groups,rounded_groups,extreme_group_sets,bits,period_indices,cfg:ProposedConfig,*,tighten_final=False,embedding_feasible=None):
    """Compatibility wrapper materializing ``PeriodDecision`` records."""
    del bits
    st=certify_spread_groups_arrays(continuous_groups,rounded_groups,extreme_group_sets,period_indices,cfg,tighten_final=tighten_final,embedding_feasible=embedding_feasible)
    n=st['period_indices'].size; extreme=st['extreme_bounds']; intervals=st['path_interval_bounds']
    out=[]
    for k in range(n):
        ex=tuple(float(x) for x in extreme[:,k]) if extreme.size else ()
        ints=tuple(float(x) for x in intervals[:,k]) if intervals.size else ()
        ef=bool(st['embedding_feasible'][k]); cert=bool(st['certified'][k])
        out.append(PeriodDecision(
            int(st['period_indices'][k]),float(st['periods'][k]),float(st['convolution_shift'][k]),
            float(st['required_margin'][k]),float(st['certificate_margin'][k]),float(st['worst_case_mse'][k]),
            cert,ef,ef,float(st['convolution_bound'][k]),ex,float(st['rounding_bound'][k]),
            float(st['rounding_shift'][k]),'embedding_infeasible' if not ef else ('certified' if cert else 'uncertified'),
            'two_extreme_piecewise_path' if bool(st['use_path'][k]) else 'generic',
            float(st['fallback_bound'][k]),ints
        ))
    return out
