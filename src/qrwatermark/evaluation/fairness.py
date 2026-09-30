from __future__ import annotations
from dataclasses import dataclass,replace
import math
import numpy as np
from .metrics import psnr
from ..proposed.method import ConvolutionCertifiedR12QIM

@dataclass(frozen=True)
class ResourceBudget:
    payload_bits:int
    host_pixels:int
    selected_blocks:int|None
    side_information_bits:int|None
    repetition:int|None


def proposed_budget(payload_bits:int,host_pixels:int,repetition:int,period_count:int=4,include_certificate_mask:bool=True)->ResourceBudget:
    bits_per_code=max(1,int(math.ceil(math.log2(period_count))))
    side=payload_bits*bits_per_code + (payload_bits if include_certificate_mask else 0) + 256
    return ResourceBudget(payload_bits,host_pixels,payload_bits*repetition,side,repetition)


def tune_baseline_to_psnr(method,host,watermark,*,key:bytes,target_psnr_db:float,scales=None):
    """Tune a baseline quantization step against the proposal's realized PSNR.

    Returns (tuned_method, embedding_result, diagnostics).  This is intentionally
    host/watermark specific so robustness is not purchased by a larger distortion
    budget than the proposal on that exact sample.
    """
    if isinstance(method,ConvolutionCertifiedR12QIM):
        emb=method.embed(host,watermark,key=key)
        return method,emb,{"target_psnr_db":float(target_psnr_db),"achieved_psnr_db":psnr(host,emb.image),"strength_scale":1.0}
    if not hasattr(method,"config") or not hasattr(method.config,"quant_step"):
        emb=method.embed(host,watermark,key=key)
        return method,emb,{"target_psnr_db":float(target_psnr_db),"achieved_psnr_db":psnr(host,emb.image),"strength_scale":1.0}
    auto_scales=scales is None
    scales=np.asarray(scales if scales is not None else np.geomspace(0.5,8.0,9),dtype=np.float64)
    base=float(method.config.quant_step); trials=[]
    import time
    total_t0=time.perf_counter()
    def trial(scale):
        tuned=method.__class__(replace(method.config,quant_step=base*float(scale)))
        t0=time.perf_counter(); emb=tuned.embed(host,watermark,key=key); embed_s=time.perf_counter()-t0; p=float(psnr(host,emb.image))
        trials.append((abs(p-float(target_psnr_db)),p,float(scale),tuned,emb,float(embed_s)))
    for scale in scales: trial(scale)
    if auto_scales:
        best=min(trials,key=lambda x:(x[0],-x[1])); center=float(best[2])
        for scale in np.linspace(max(0.05,0.8*center),1.2*center,5):
            if not np.any(np.isclose(scales,scale,rtol=0,atol=1e-12)): trial(float(scale))
    _,p,scale,tuned,emb,embed_s=min(trials,key=lambda x:(x[0],-x[1]))
    return tuned,emb,{"target_psnr_db":float(target_psnr_db),"achieved_psnr_db":p,"strength_scale":scale,"quant_step":float(tuned.config.quant_step),"embed_seconds":embed_s,"tuning_seconds":float(time.perf_counter()-total_t0),"psnr_error_db":float(p-target_psnr_db)}
