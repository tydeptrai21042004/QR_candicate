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
    block_size:int=2; channel:int=0; watermark_size:int=64; arnold_iterations:int=10; repetition:int=5
    period_candidates:tuple[float,...]=(24.0,28.0,32.0,36.0)
    max_group_mse:float=220.0
    target_psnr_db:float=50.0
    convolution_kernel_size:int=3
    convolution_gaussian_sigmas:tuple[float,...]=(0.50,0.75)
    convolution_safety_factor:float=1.0
    additive_feature_budget:float=1.0
    rounding_feature_budget:float=0.5
    nlm:NLMConfig=field(default_factory=NLMConfig)
    def validate(self):
        if self.block_size!=2: raise ValueError("MC-CCQR spread-QIM is derived for 2x2 QR blocks")
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
    def as_dict(self)->dict[str,Any]: return asdict(self)

@dataclass
class BaselineConfig:
    block_size:int=4; channel:int=0; watermark_size:int=64; arnold_iterations:int=10; quant_step:float=8.0; threshold:float=0.04

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
