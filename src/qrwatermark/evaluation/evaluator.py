from __future__ import annotations
from time import perf_counter
from typing import Any
import numpy as np
from ..attacks import apply_attack
from ..core.interfaces import WatermarkMethod
from ..core.types import BenchmarkRecord,EmbeddingResult
from .metrics import ber,nc,psnr,ssim
from ..utils.watermark import arnold_transform,bits_from_watermark

_RANDOM_ATTACKS={"gaussian_noise","salt_pepper","speckle_noise","random_pixel_dropout","occlusion"}


def embed_once(method:WatermarkMethod,host:np.ndarray,watermark:np.ndarray,*,key:bytes)->tuple[EmbeddingResult,float]:
    t0=perf_counter(); emb=method.embed(host,watermark,key=key); return emb,float(perf_counter()-t0)


def evaluate_embedded(method:WatermarkMethod,host:np.ndarray,watermark:np.ndarray,emb:EmbeddingResult,*,embed_seconds:float,key:bytes,host_name:str,watermark_name:str,attack_name:str="clean",attack_params:dict[str,Any]|None=None,seed:int|None=None):
    attack_params=dict(attack_params or {})
    if seed is not None and attack_name in _RANDOM_ATTACKS: attack_params.setdefault("seed",seed)
    attacked=apply_attack(attack_name,emb.image,**attack_params)
    t0=perf_counter(); ext=method.extract(attacked,key=key,side_info=emb.side_info,watermark_shape=watermark.shape[:2]); extract_s=perf_counter()-t0
    ep=psnr(host,emb.image); es=ssim(host,emb.image); ap=psnr(host,attacked); ass=ssim(host,attacked)
    overhead=emb.metadata.get("side_information_overhead_bits",{}) if emb.metadata else {}
    certified_ber=None; uncertified_ber=None
    if emb.side_info is not None and "packed_certified_mask" in emb.side_info:
        from ..proposed.side_info import unpack_certified_mask
        mask=unpack_certified_mask(emb.side_info,key)
        iters=int(emb.side_info.get("metadata",{}).get("arnold_iterations",0))
        truth=bits_from_watermark(arnold_transform(watermark,iters))
        recovered=bits_from_watermark(arnold_transform(ext.watermark,iters))
        if mask.size==truth.size:
            if np.any(mask): certified_ber=float(np.mean(truth[mask]!=recovered[mask]))
            if np.any(~mask): uncertified_ber=float(np.mean(truth[~mask]!=recovered[~mask]))
    logical_side_bits=(int(overhead.get("period_code_bits",0))+int(overhead.get("certificate_mask_bits",0))+int(overhead.get("authentication_bits",0)) if overhead else (int(emb.metadata.get("side_information_bits",0)) if emb.metadata else 0))
    serialized_side_bits=(int(overhead.get("total_serialized_bits",logical_side_bits)) if overhead else logical_side_bits)
    record=BenchmarkRecord(
        method=method.name,host=host_name,watermark=watermark_name,attack=attack_name,
        attack_params=str(attack_params),seed=seed,psnr=ep,ssim=es,
        embedding_psnr=ep,embedding_ssim=es,attacked_psnr=ap,attacked_ssim=ass,
        nc=nc(watermark,ext.watermark),ber=ber(watermark,ext.watermark),
        embed_seconds=float(embed_seconds),extract_seconds=float(extract_s),confidence=ext.confidence,
        side_information_bits=logical_side_bits,side_information_serialized_bits=serialized_side_bits,
        certified_fraction=float(emb.metadata.get("certified_fraction")) if emb.metadata and "certified_fraction" in emb.metadata else None,
        certified_ber=certified_ber,uncertified_ber=uncertified_ber,
    )
    return record,{"watermarked":emb.image,"attacked":attacked,"extracted":ext.watermark,"embedding_metadata":emb.metadata,"extraction_metadata":ext.metadata,"side_info":emb.side_info}


def evaluate_once(method:WatermarkMethod,host:np.ndarray,watermark:np.ndarray,*,key:bytes,host_name:str,watermark_name:str,attack_name:str="clean",attack_params:dict[str,Any]|None=None,seed:int|None=None):
    emb,embed_s=embed_once(method,host,watermark,key=key)
    return evaluate_embedded(method,host,watermark,emb,embed_seconds=embed_s,key=key,host_name=host_name,watermark_name=watermark_name,attack_name=attack_name,attack_params=attack_params,seed=seed)
