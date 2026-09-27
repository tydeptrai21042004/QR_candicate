from __future__ import annotations
import numpy as np

def validate_kernel(kernel,*,atol=1e-10):
    k=np.asarray(kernel,dtype=np.float64)
    if k.ndim!=2: raise ValueError("kernel must be 2-D")
    if np.any(k < -atol): raise ValueError("certified kernels must be non-negative")
    if abs(float(k.sum())-1.0)>atol: raise ValueError("certified kernels must sum to one")
    return k

def convex_mixture(kernels,coefficients):
    if not kernels: raise ValueError("kernels must be non-empty")
    c=np.asarray(coefficients,dtype=np.float64).ravel()
    if c.size!=len(kernels) or np.any(c < -1e-12) or abs(float(c.sum())-1.0)>1e-10: raise ValueError("coefficients must be non-negative and sum to one")
    ks=[validate_kernel(k) for k in kernels]; shape=ks[0].shape
    if any(k.shape!=shape for k in ks): raise ValueError("all kernels must have equal shape")
    return np.tensordot(c,np.stack(ks),axes=(0,0))

def random_convex_mixtures(kernels,count,seed=0):
    rng=np.random.default_rng(seed)
    return [convex_mixture(kernels,rng.dirichlet(np.ones(len(kernels)))) for _ in range(int(count))]
