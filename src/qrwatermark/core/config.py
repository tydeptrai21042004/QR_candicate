from __future__ import annotations
from dataclasses import asdict,dataclass,field
from pathlib import Path
from typing import Any
import yaml

@dataclass
class NLMConfig:
    enabled:bool=False; compare_always:bool=False; mild_h:float=4.0; mild_template:int=3; mild_search:int=11; strong_h:float=8.0; strong_template:int=5; strong_search:int=17

@dataclass
class ProposedConfig:
    # ``design`` keeps the legacy research implementation reproducible while
    # allowing the default YAML to use the cleaner single-carrier formulation.
    design:str="legacy_v1"
    block_size:int=2; channel:int=0; watermark_size:int=64; arnold_iterations:int=10; repetition:int=5
    qim_period:float=64.0
    carrier_pool:int=4
    carrier_selector_policy:str="min_energy"
    # v3 fully-blind parameters.  The default 1/8 margin is symmetric inside
    # each half-period and leaves a nonzero safe interval.
    blind_margin_ratio:float=0.125
    blind_projection:str="safe_set"
    # v4 fully-blind coupled Q/R parameters. One 2x2 block carries one payload
    # bit through two observations inside the same QR factorization.
    coupled_q_period:float=0.20
    coupled_r_period:float=0.48
    coupled_q_margin_ratio:float=0.08
    coupled_r_margin_ratio:float=0.08
    coupled_q_weight:float=0.5
    coupled_r_weight:float=1.0
    coupled_energy_tau:float=16.0
    compute_certificate:bool=True
    store_certificate_mask:bool=False
    period_candidates:tuple[float,...]=(24.0,28.0,32.0,36.0)
    max_group_mse:float=220.0
    target_psnr_db:float=50.0
    convolution_kernel_size:int=3
    convolution_gaussian_sigmas:tuple[float,...]=(0.50,0.75)
    convolution_safety_factor:float=1.0
    additive_feature_budget:float=1.0
    rounding_feature_budget:float=0.0  # deprecated; deterministic rounding bound is computed per group
    certificate_max_passes:int=4
    certificate_path_subdivisions:int=8
    certificate_final_tighten:bool=True
    nlm:NLMConfig=field(default_factory=NLMConfig)
    def validate(self):
        if self.block_size!=2: raise ValueError("proposal is derived for 2x2 QR blocks")
        if self.design not in {"legacy_v1","elegant_v2","blind_v3","blind_v4"}: raise ValueError("design must be legacy_v1, elegant_v2, blind_v3, or blind_v4")
        if self.qim_period<=0: raise ValueError("qim_period must be positive")
        if self.carrier_pool<1 or self.carrier_pool>255: raise ValueError("carrier_pool must be in [1,255]")
        if self.carrier_selector_policy not in {"min_energy","min_rounded_sse","first"}: raise ValueError("unsupported carrier_selector_policy")
        if not (0.0 < float(self.blind_margin_ratio) < 0.25): raise ValueError("blind_margin_ratio must lie in (0,0.25)")
        if self.blind_projection not in {"safe_set","center"}: raise ValueError("blind_projection must be safe_set or center")
        if self.coupled_q_period<=0 or self.coupled_r_period<=0: raise ValueError("coupled Q/R periods must be positive")
        if not (0.0<float(self.coupled_q_margin_ratio)<0.25): raise ValueError("coupled_q_margin_ratio must lie in (0,0.25)")
        if not (0.0<float(self.coupled_r_margin_ratio)<0.25): raise ValueError("coupled_r_margin_ratio must lie in (0,0.25)")
        if self.coupled_q_weight<0 or self.coupled_r_weight<=0: raise ValueError("coupled fusion weights must be non-negative with positive R weight")
        if self.coupled_energy_tau<0: raise ValueError("coupled_energy_tau must be non-negative")
        if self.design in {"blind_v3","blind_v4"} and self.store_certificate_mask: raise ValueError("fully blind designs never store a certificate mask or any other side information")
        if self.repetition<1: raise ValueError("repetition must be positive")
        periods=tuple(float(x) for x in self.period_candidates)
        if not periods or any(x<=0 for x in periods): raise ValueError("period_candidates must contain positive values")
        if sorted(set(periods))!=list(periods): raise ValueError("period_candidates must be strictly increasing and unique")
        if len(periods)>256: raise ValueError("at most 256 period candidates are supported")
        if self.max_group_mse<=0 or self.target_psnr_db<=0: raise ValueError("distortion/PSNR targets must be positive")
        if self.convolution_kernel_size<1 or self.convolution_kernel_size%2==0: raise ValueError("convolution_kernel_size must be positive odd")
        if any(float(s)<=0 for s in self.convolution_gaussian_sigmas): raise ValueError("all convolution sigmas must be positive")
        if self.convolution_safety_factor<1.0: raise ValueError("convolution_safety_factor must be >=1")
        if self.additive_feature_budget<0 or self.rounding_feature_budget<0: raise ValueError("feature budgets must be non-negative")
        if self.certificate_max_passes<1: raise ValueError("certificate_max_passes must be positive")
        if self.certificate_path_subdivisions<1: raise ValueError("certificate_path_subdivisions must be positive")
    def as_dict(self)->dict[str,Any]: return asdict(self)

@dataclass
class BaselineConfig:
    block_size:int=4; channel:int=0; watermark_size:int=64; arnold_iterations:int=10; quant_step:float=8.0; threshold:float=0.04; gamma:float=3.25

def load_yaml(path:str|Path)->dict[str,Any]:
    with open(path,'r',encoding='utf-8') as f: obj=yaml.safe_load(f)
    return obj or {}

def proposed_from_dict(data):
    d=dict(data); nlm=d.pop('nlm',None)
    if 'period_candidates' in d: d['period_candidates']=tuple(float(x) for x in d['period_candidates'])
    if 'convolution_gaussian_sigmas' in d: d['convolution_gaussian_sigmas']=tuple(float(x) for x in d['convolution_gaussian_sigmas'])
    cfg=ProposedConfig(**d)
    if nlm is not None: cfg.nlm=NLMConfig(**nlm)
    cfg.validate(); return cfg

def baseline_from_dict(data): return BaselineConfig(**data)
