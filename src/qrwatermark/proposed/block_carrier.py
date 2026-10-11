"""Deterministic keyed non-overlapping 4x4 carrier selection.

The SHA-256/PCG64 domain is unchanged from the original recovered block mapper
for reproducibility. An LRU cache avoids rebuilding the same permutation for
frame sequences and repeated trials. This is an implementation optimization,
NOT a new watermarking mathematical contribution.
"""
from __future__ import annotations
import hashlib
from functools import lru_cache
import numpy as np

@lru_cache(maxsize=16)
def _positions(h:int,w:int,count:int,key:bytes):
    rows,cols=h//4,w//4
    capacity=rows*cols
    if count<0 or count>capacity:
        raise ValueError(f'requested {count} patches; capacity is {capacity}')
    payload=(b'qrwatermark-recovered-block-carrier-v1\0'+
             len(key).to_bytes(4,'big')+key+
             h.to_bytes(8,'big')+w.to_bytes(8,'big'))
    seed=int.from_bytes(hashlib.sha256(payload).digest()[:16],'big')
    rng=np.random.Generator(np.random.PCG64(seed))
    ix=rng.permutation(capacity)[:count]
    rr=(ix//cols*4).astype(np.intp)
    cc=(ix%cols*4).astype(np.intp)
    rr.flags.writeable=False;cc.flags.writeable=False
    return rr,cc

def keyed_blocks(image_shape,count,key):
    h,w=int(image_shape[0]),int(image_shape[1])
    if not isinstance(key,bytes):raise TypeError('key must be bytes')
    return _positions(h,w,int(count),key)

def _gather_blocks(channel,rr,cc):
    x=np.asarray(channel)
    if x.ndim!=2:raise ValueError('expected 2D channel')
    rr=np.asarray(rr,dtype=np.intp).ravel()
    cc=np.asarray(cc,dtype=np.intp).ravel()
    if rr.shape!=cc.shape:raise ValueError('row and column index mismatch')
    off=np.arange(4,dtype=np.intp)
    return x[rr[:,None,None]+off[None,:,None],cc[:,None,None]+off[None,None,:]].copy()
