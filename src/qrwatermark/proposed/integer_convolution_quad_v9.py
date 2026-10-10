"""Four independent 2x2 fixed-direction QR carriers in a keyed 4x4 patch.

All four integer low-pass sums share the keyed block lookup but no anchor,
normalization, or pixels.  One fully blind decoder is used for all attacks.
"""
from __future__ import annotations
import numpy as np
from .fused_convqr_v6 import _gather_blocks, keyed_blocks_v6
from .integer_convolution_v7 import _check_image_and_shape, _distribute_exact, _lattice_centres
from ..utils.watermark import scrambled_bits_from_watermark, watermark_from_scrambled_bits


def _to_cells(patches):
    # [N,4,4] -> [N,4,4]  (4 disjoint 2x2 cells, 4 pixels each)
    return patches.reshape(-1,2,2,2,2).transpose(0,1,3,2,4).reshape(-1,4,4)


def _to_patches(cells):
    return cells.reshape(-1,2,2,2,2).transpose(0,1,3,2,4).reshape(-1,4,4)


def embed_integer_conv_quad_image(host,watermark,key,cfg):
    cfg.validate()
    image=np.asarray(host)
    wm=np.asarray(watermark,dtype=np.uint8)
    _check_image_and_shape(image,wm.shape)
    expected=(cfg.watermark_size,cfg.watermark_size)
    if wm.shape!=expected:raise ValueError(f'expected watermark shape {expected}')
    bits=scrambled_bits_from_watermark(wm,cfg.arnold_iterations).ravel()
    if bits.size%4:raise ValueError('four-carrier QIM requires payload divisible by four')
    rr,cc=keyed_blocks_v6(image.shape[:2],bits.size//4,key)
    patches=_gather_blocks(image[:,:,cfg.channel],rr,cc)
    vals=_to_cells(patches).reshape(-1,4)
    sums=vals.astype(np.int16).sum(axis=1).astype(np.int32)
    targets=_lattice_centres(sums,bits,int(cfg.integer_conv4_period),max_sum=1020)
    updated=_distribute_exact(vals,targets).reshape(-1,4,4)
    modified=_to_patches(updated)
    result=image.copy()
    channel=result[:,:,cfg.channel];h,w=channel.shape
    channel[:h//4*4,:w//4*4].reshape(h//4,4,w//4,4)[rr//4,:,cc//4,:]=modified
    return result,None,{
        'method':'blind_integer_convolution_quad_v9_experimental',
        'fully_blind':True,'side_information_bits':0,'payload_bits':int(bits.size),
        'qim_period':int(cfg.integer_conv4_period),'clean_integer_failures':0,
        'convolution':'four independent 2x2 low-pass carriers per 4x4 block',
        'runtime_path':'four_integer_sums_fixed_direction_qr',
    }


def extract_integer_conv_quad_image(image,key,cfg,watermark_shape=None):
    cfg.validate()
    x=np.asarray(image)
    expected=(cfg.watermark_size,cfg.watermark_size)
    if watermark_shape is not None and tuple(watermark_shape)!=expected:
        raise ValueError(f'expected watermark shape {expected}')
    _check_image_and_shape(x,expected)
    count=expected[0]*expected[1]
    if count%4:raise ValueError('four-carrier QIM requires payload divisible by four')
    rr,cc=keyed_blocks_v6(x.shape[:2],count//4,key)
    p=_gather_blocks(x[:,:,cfg.channel],rr,cc)
    sums=_to_cells(p).reshape(-1,4).astype(np.int16).sum(axis=1)
    period=int(cfg.integer_conv4_period); rem=sums%period
    bits=(rem>=period//2).astype(np.uint8)
    z=rem%(period//2)
    confidence=float(np.mean(np.minimum(z,period//2-z)/(period/4)))
    return watermark_from_scrambled_bits(bits,expected[0],cfg.arnold_iterations),confidence,{
        'method':'blind_integer_convolution_quad_v9_experimental',
        'fully_blind':True,'original_host_used':False,'side_information_used':False,
        'runtime_path':'four_integer_sums_fixed_direction_qr',
    }
