"""Minimal configuration for one proposed method plus published baselines."""
from __future__ import annotations
from dataclasses import dataclass,asdict
from pathlib import Path
from typing import Any
import yaml

@dataclass
class ProposedConfig:
    design:str='green_quad_min_energy'
    block_size:int=4
    channel:int=1
    watermark_size:int=64
    arnold_iterations:int=0
    integer_conv4_period:int=66
    projection:str='optimal'
    ablation_mode:bool=False
    compute_certificate:bool=False
    store_certificate_mask:bool=False
    def validate(self):
        if self.design!='green_quad_min_energy':raise ValueError('Only green_quad_min_energy proposal is supported')
        if self.block_size!=4:raise ValueError('carrier block_size must be 4')
        if self.watermark_size<4 or self.watermark_size%2:raise ValueError('watermark_size must be even >=4')
        if type(self.integer_conv4_period)!=int or self.integer_conv4_period<4 or self.integer_conv4_period%2:raise ValueError('integer_conv4_period must be even >=4')
        if self.channel not in (0,1,2):raise ValueError('channel is BGR index 0/1/2')
        if self.arnold_iterations<0:raise ValueError('arnold_iterations must be nonnegative')
        if self.projection not in ('optimal','greedy_control'):raise ValueError('unknown integer projection')
        if self.store_certificate_mask:raise ValueError('side information is not supported')
        if not self.ablation_mode and (self.channel!=1 or self.arnold_iterations!=0 or self.projection!='optimal'):
            raise ValueError('production method uses green channel, no Arnold, exact projection; use ablation_mode=True for controls')
    def as_dict(self)->dict[str,Any]:return asdict(self)

@dataclass
class BaselineConfig:
    block_size:int=4;channel:int=0;watermark_size:int=64;arnold_iterations:int=10;quant_step:float=8.0;threshold:float=0.04;gamma:float=3.25

def load_yaml(path:str|Path)->dict[str,Any]:
    with open(path,'r',encoding='utf-8') as f:obj=yaml.safe_load(f)
    return obj or {}

def proposed_from_dict(data):
    cfg=ProposedConfig(**data);cfg.validate();return cfg

def baseline_from_dict(data):return BaselineConfig(**data)
