from __future__ import annotations
import cv2
import numpy as np
from ..core.config import ProposedConfig
from .r_branch import r12_value
from .spread_qim import spread_decision,unit_spread_weights


def _decode_groups_reference(channel,positions,periods,repetition,block_size):
    """Legacy scalar implementation retained as an equivalence oracle."""
    periods=np.asarray(periods,dtype=np.float64).ravel(); L=periods.size
    if len(positions)!=L*repetition: raise ValueError("positions count mismatch")
    w=unit_spread_weights(repetition); bits=np.zeros(L,dtype=np.uint8); confs=np.zeros(L)
    for k,p in enumerate(periods):
        vals=[]
        for j in range(repetition):
            r,c=positions[k*repetition+j]; vals.append(r12_value(channel[r:r+block_size,c:c+block_size].astype(np.float64)))
        bit,_,confidence=spread_decision(np.asarray(vals),float(p),w)
        bits[k]=bit; confs[k]=confidence
    return bits,float(confs.mean()),confs


def _selected_r12_vectorized(channel, positions, block_size):
    """Extract canonical r12 for all selected 2x2 blocks in one NumPy batch."""
    if int(block_size) != 2:
        raise ValueError("vectorized edge decoder is derived for 2x2 blocks")
    x=np.asarray(channel,dtype=np.float64)
    pos=np.asarray(positions,dtype=np.int64)
    if pos.ndim!=2 or pos.shape[1]!=2:
        raise ValueError("positions must contain (row,col) pairs")
    rr=pos[:,0]; cc=pos[:,1]
    a0=x[rr,cc]; a1=x[rr+1,cc]
    b0=x[rr,cc+1]; b1=x[rr+1,cc+1]
    norm=np.sqrt(a0*a0+a1*a1)
    vals=np.empty(norm.size,dtype=np.float64)
    good=norm>1e-12
    vals[good]=(a0[good]*b0[good]+a1[good]*b1[good])/norm[good]
    # Degenerate blocks are rare in natural imagery.  Preserve exact canonical
    # QR semantics only for those entries rather than penalizing every carrier.
    if np.any(~good):
        bad=np.flatnonzero(~good)
        for idx in bad:
            r=int(rr[idx]); c=int(cc[idx])
            vals[idx]=r12_value(x[r:r+2,c:c+2])
    return vals


def decode_groups(channel,positions,periods,repetition,block_size):
    """Vectorized hardware-form decoder with the same QIM hard decisions.

    The hot path contains only selected pixel gathers, multiply-adds, square
    roots, a 5-tap spread accumulation, modulo by one of four constant periods,
    and comparisons.  This mirrors the intended SIMD/NEON/FPGA datapath.
    """
    periods=np.asarray(periods,dtype=np.float64).ravel(); L=periods.size
    rep=int(repetition)
    if len(positions)!=L*rep: raise ValueError("positions count mismatch")
    vals=_selected_r12_vectorized(channel,positions,block_size).reshape(L,rep)
    # unit_spread_weights(rep) is constant 1/sqrt(rep); matrix multiply keeps
    # the software reference close to the mathematical dot product.
    w=unit_spread_weights(rep)
    stats=vals@w
    phase=np.mod(stats,periods)
    half=0.5*periods; quarter=0.25*periods
    bits=(phase>=half).astype(np.uint8)
    confs=np.minimum(np.minimum(phase,np.abs(phase-half)),periods-phase)/quarter
    confs=np.clip(confs,0.0,1.0)
    return bits,float(confs.mean()),confs.astype(np.float64,copy=False)


def nlm_versions(channel,cfg:ProposedConfig):
    raw=np.asarray(channel,dtype=np.float64); versions=[('raw',raw)]
    if not cfg.nlm.enabled: return versions
    raw8=np.clip(np.rint(raw),0,255).astype(np.uint8)
    mild=cv2.fastNlMeansDenoising(raw8,None,h=float(cfg.nlm.mild_h),templateWindowSize=cfg.nlm.mild_template,searchWindowSize=cfg.nlm.mild_search)
    strong=cv2.fastNlMeansDenoising(raw8,None,h=float(cfg.nlm.strong_h),templateWindowSize=cfg.nlm.strong_template,searchWindowSize=cfg.nlm.strong_search)
    return versions+[('nlm_mild',mild),('nlm_strong',strong)]
