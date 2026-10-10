"""Experimental two-carrier fixed-direction QR / integer-convolution method.

One keyed 4x4 block holds two disjoint eight-pixel low-pass carriers:
left two columns and right two columns. Two QIM bits share the block
lookup but NEVER share a noise-sensitive QR anchor.  Both use the same
attack-independent one-sum extraction formula.
"""
from __future__ import annotations
import numpy as np
from .block_carrier import _gather_blocks, keyed_blocks_v6
from .integer_convolution_v7 import _check_image_and_shape, _distribute_exact, _lattice_centres
from ..utils.watermark import scrambled_bits_from_watermark, watermark_from_scrambled_bits


def _to_carriers(patches):
    # [N,4,4] -> [N,2,8] via left/right disjoint eight-pixel supports
    return patches.reshape(-1,4,2,2).transpose(0,2,1,3).reshape(-1,2,8)


def _to_patches(carriers):
    return carriers.reshape(-1,2,4,2).transpose(0,2,1,3).reshape(-1,4,4)


def embed_integer_conv_pair_image(host, watermark, key, cfg):
    cfg.validate()
    image=np.asarray(host)
    wm=np.asarray(watermark,dtype=np.uint8)
    _check_image_and_shape(image,wm.shape)
    expected=(cfg.watermark_size,cfg.watermark_size)
    if wm.shape!=expected:raise ValueError(f'expected watermark shape {expected}')
    bits=scrambled_bits_from_watermark(wm,cfg.arnold_iterations).ravel()
    if bits.size%2:raise ValueError('two-carrier QIM requires even payload bits')
    rr,cc=keyed_blocks_v6(image.shape[:2],bits.size//2,key)
    patches=_gather_blocks(image[:,:,cfg.channel],rr,cc)
    vals=_to_carriers(patches).reshape(-1,8)
    sums=vals.astype(np.int16).sum(axis=1).astype(np.int32)
    targets=_lattice_centres(sums,bits,int(cfg.integer_conv_period))
    updated=_distribute_exact(vals,targets).reshape(-1,2,8)
    modified=_to_patches(updated)
    output=image.copy()
    channel=output[:,:,cfg.channel]
    h,w=channel.shape
    channel[:h//4*4,:w//4*4].reshape(h//4,4,w//4,4)[rr//4,:,cc//4,:]=modified
    return output,None,{
        'method':'blind_integer_convolution_pair_v8_experimental',
        'fully_blind':True,'side_information_bits':0,'payload_bits':int(bits.size),
        'qim_period':int(cfg.integer_conv_period),'clean_integer_failures':0,
        'convolution':'two nonoverlapping 8-pixel low-pass carriers per 4x4 block',
        'runtime_path':'dual_integer_sum_fixed_direction_qr',
    }


def extract_integer_conv_pair_image(image,key,cfg,watermark_shape=None):
    cfg.validate()
    x=np.asarray(image)
    expected=(cfg.watermark_size,cfg.watermark_size)
    if watermark_shape is not None and tuple(watermark_shape)!=expected:
        raise ValueError(f'expected watermark shape {expected}')
    _check_image_and_shape(x,expected)
    count=expected[0]*expected[1]
    if count%2:raise ValueError('two-carrier QIM requires even payload bits')
    rr,cc=keyed_blocks_v6(x.shape[:2],count//2,key)
    patches=_gather_blocks(x[:,:,cfg.channel],rr,cc)
    sums=_to_carriers(patches).reshape(-1,8).astype(np.int16).sum(axis=1)
    period=int(cfg.integer_conv_period)
    residue=sums%period
    bits=(residue>=period//2).astype(np.uint8)
    r=residue%(period//2)
    confidence=float(np.mean(np.minimum(r,period//2-r)/(period/4)))
    watermark=watermark_from_scrambled_bits(bits,expected[0],cfg.arnold_iterations)
    return watermark,confidence,{
       'method':'blind_integer_convolution_pair_v8_experimental',
       'fully_blind':True,'original_host_used':False,'side_information_used':False,
       'runtime_path':'dual_integer_sum_fixed_direction_qr',
    }
