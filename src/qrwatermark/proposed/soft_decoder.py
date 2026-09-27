from __future__ import annotations
import cv2
import numpy as np
from ..core.config import ProposedConfig
from .r_branch import r12_value
from .spread_qim import spread_llr,unit_spread_weights

def decode_groups(channel,positions,periods,repetition,block_size):
    periods=np.asarray(periods,dtype=np.float64).ravel(); L=periods.size
    if len(positions)!=L*repetition: raise ValueError("positions count mismatch")
    w=unit_spread_weights(repetition); llrs=np.zeros(L)
    for k,p in enumerate(periods):
        vals=[]
        for j in range(repetition):
            r,c=positions[k*repetition+j]; vals.append(r12_value(channel[r:r+block_size,c:c+block_size].astype(np.float64)))
        llrs[k]=spread_llr(np.asarray(vals),float(p),w)
    bits=(llrs>=0).astype(np.uint8); conf=np.abs(llrs)
    return bits,float(conf.mean()),conf

def nlm_versions(channel,cfg:ProposedConfig):
    raw=channel.astype(np.uint8); versions=[('raw',raw)]
    if not cfg.nlm.enabled: return versions
    mild=cv2.fastNlMeansDenoising(raw,None,h=float(cfg.nlm.mild_h),templateWindowSize=cfg.nlm.mild_template,searchWindowSize=cfg.nlm.mild_search)
    strong=cv2.fastNlMeansDenoising(raw,None,h=float(cfg.nlm.strong_h),templateWindowSize=cfg.nlm.strong_template,searchWindowSize=cfg.nlm.strong_search)
    return versions+[('nlm_mild',mild),('nlm_strong',strong)]
