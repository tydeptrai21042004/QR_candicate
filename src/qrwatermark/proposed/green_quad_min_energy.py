"""Research candidate: four-cell integer convolution with provably optimal
per-carrier squared-error projection for uint8 pixels.

Fixed green (OpenCV BGR channel 1); QIM step 66 by default. No host,
side information, repetitions, or pilots. This is NOT a new QR transform.
"""
from __future__ import annotations
import numpy as np
from .block_carrier import _gather_blocks,keyed_blocks
from ..utils.watermark import scrambled_bits_from_watermark,watermark_from_scrambled_bits



def _check_image_and_shape(image, wm_shape):
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError('expected HxWx3 uint8 BGR image')
    if len(wm_shape) != 2 or wm_shape[0] != wm_shape[1]:
        raise ValueError('watermark must be square')
    if image.shape[0] // 4 * (image.shape[1] // 4) * 4 < wm_shape[0]*wm_shape[1]:
        raise ValueError('host has insufficient 4x4 block capacity')

def _to_cells(p):
    return p.reshape(-1,2,2,2,2).transpose(0,1,3,2,4).reshape(-1,4,4)

def _to_patches(c):
    return c.reshape(-1,2,2,2,2).transpose(0,1,3,2,4).reshape(-1,4,4)

def _lattice_centres(sums,bits,period,max_sum=1020):
    base = period//4 + (period//2)*bits.astype(np.int32)
    k = np.rint((sums.astype(np.float64)-base)/period).astype(np.int32)
    return base+period*np.maximum(0,np.minimum(k,(max_sum-base)//period))

def optimal_integer_projection(values,targets,stats_out=None):
    """Global integer least-squares projection for a FIXED sum target, 0<=y<=255.

    Discrete water filling: F(q)=sum(clip(x+q,0,255)); choose q such that
    F(q)<=target<F(q+1), then give +1 to exactly target-F(q)
    non-saturated positions. Convex marginal-cost exchange proves optimum.
    Vectorized O(n) unconstrained fast path, O(n log256) fallback.
    """
    a=np.asarray(values)
    if a.ndim!=2 or a.dtype.kind not in 'iu' or np.any(a<0) or np.any(a>255):
        raise ValueError('expected integer array in [0,255]')
    n=a.shape[1]
    target=np.asarray(targets,dtype=np.int32).ravel()
    if target.shape[0]!=a.shape[0] or np.any(target<0) or np.any(target>255*n):
        raise ValueError('infeasible target')
    x=a.astype(np.int32,copy=False)
    d=target-x.sum(axis=1)
    q,r=np.divmod(d,n)
    z=x+q[:,None]+(np.arange(n)[None,:]<r[:,None])
    bad=np.flatnonzero(np.any((z<0)|(z>255),axis=1))
    if bad.size:
        xb=x[bad]; y=target[bad]
        lo=np.full(len(bad),-255,dtype=np.int32)
        hi=np.full(len(bad),255,dtype=np.int32)
        # lower-bound q with F(q)<=target, final lo=q
        for _ in range(9):
            mid=(lo+hi+1)//2
            f=np.clip(xb+mid[:,None],0,255).sum(axis=1)
            ok=f<=y
            lo=np.where(ok,mid,lo)
            hi=np.where(ok,hi,mid-1)
        base=np.clip(xb+lo[:,None],0,255)
        remaining=y-base.sum(axis=1)
        unclipped=xb+lo[:,None]
        can=(unclipped>=0)&(unclipped<255)
        # For target==n*255, q=255 works and remaining=0.
        rank=np.cumsum(can,axis=1)
        base+=((can)&(rank<=remaining[:,None])).astype(np.int32)
        z[bad]=base
    if stats_out is not None:
        stats_out.update(fallback_cells=int(len(bad)),total_cells=int(len(x)),correction_failures=0)
    if np.any(z.sum(axis=1)!=target) or np.any(z<0) or np.any(z>255):
        raise AssertionError('exact integer projection failed')
    return z.astype(np.uint8)

def greedy_integer_projection(values,targets):
    """Deterministic feasible nonoptimal control, used ONLY for an ablation.

    It updates the first eligible pixel at each unit step; intentionally not
    an online production algorithm. Contrast its squared distortion with the
    optimal water-filling solver. No BER benefit is assumed.
    """
    x=np.asarray(values,dtype=np.int32)
    t=np.asarray(targets,dtype=np.int32)
    z=x.copy()
    for i in range(len(x)):
        residual=int(t[i]-z[i].sum())
        for j in range(z.shape[1]):
            if residual==0:break
            move=(min(255-z[i,j],residual) if residual>0 else
                  max(-z[i,j],residual))
            z[i,j]+=move
            residual-=move
        if residual:raise AssertionError('feasible integer target not attained')
    return z.astype(np.uint8)

def embed(host,watermark,key,period=66,arnold_iterations=0,watermark_size=64,channel=1,projection='optimal'):
    image=np.asarray(host);wm=np.asarray(watermark)
    _check_image_and_shape(image,wm.shape)
    if wm.shape!=(watermark_size,watermark_size):raise ValueError('wrong watermark shape')
    if wm.size%4:raise ValueError('watermark payload must be divisible by four')
    if period<4 or period%2:raise ValueError('period must be even >=4')
    if channel not in (0,1,2):raise ValueError('channel must be BGR index 0,1,2')
    bits=scrambled_bits_from_watermark(wm,arnold_iterations).ravel()
    rr,cc=keyed_blocks(image.shape[:2],bits.size//4,key)
    patches=_gather_blocks(image[:,:,channel],rr,cc)
    v=_to_cells(patches).reshape(-1,4)
    sums=v.astype(np.int16).sum(axis=1).astype(np.int32)
    targets=_lattice_centres(sums,bits,period,max_sum=1020)
    proj_stats={}
    if projection == 'optimal':
        changed=optimal_integer_projection(v,targets,stats_out=proj_stats)
    elif projection == 'greedy_control':
        changed=greedy_integer_projection(v,targets)
    else:
        raise ValueError('invalid projection ablation')
    updated=_to_patches(changed.reshape(-1,4,4))
    result=image.copy();plane=result[:,:,channel];h,w=plane.shape
    plane[:h//4*4,:w//4*4].reshape(h//4,4,w//4,4)[rr//4,:,cc//4,:]=updated
    return result,None,{'method':'green_quad_min_energy','period':period,'channel':channel,'fully_blind':True,'side_information_bits':0,'payload_bits':bits.size,'projection':'fixed_target_globally_optimal_integer_L2',**proj_stats}

def extract(image,key,period=66,arnold_iterations=0,watermark_shape=(64,64),watermark_size=64,channel=1):
    x=np.asarray(image);shape=(watermark_size,watermark_size);_check_image_and_shape(x,shape)
    if tuple(watermark_shape)!=shape:raise ValueError('watermark shape mismatch')
    if period<4 or period%2:raise ValueError('invalid period')
    if channel not in (0,1,2):raise ValueError('channel must be BGR index 0,1,2')
    rr,cc=keyed_blocks(x.shape[:2],watermark_size*watermark_size//4,key)
    p=_gather_blocks(x[:,:,channel],rr,cc)
    sums=_to_cells(p).reshape(-1,4).astype(np.int16).sum(axis=1)
    bits=((sums%period)>=period//2).astype(np.uint8)
    return watermark_from_scrambled_bits(bits,watermark_size,arnold_iterations),None,{'fully_blind':True,'original_host_used':False,'side_information_used':False,'method':'green_quad_min_energy'}


def embed_green_quad_min_energy_image(host,watermark,key,cfg):
    cfg.validate()
    return embed(host,watermark,key,int(cfg.integer_conv4_period),int(cfg.arnold_iterations),int(cfg.watermark_size),int(cfg.channel),cfg.projection)

def extract_green_quad_min_energy_image(image,key,cfg,watermark_shape=None):
    cfg.validate()
    expected=(int(cfg.watermark_size),)*2
    return extract(image,key,int(cfg.integer_conv4_period),int(cfg.arnold_iterations),watermark_shape or expected,int(cfg.watermark_size),int(cfg.channel))
