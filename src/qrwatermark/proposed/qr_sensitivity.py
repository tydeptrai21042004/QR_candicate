from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .r_branch import r12_value
from .spread_qim import unit_spread_weights


@dataclass(frozen=True)
class R12PerturbationBound:
    actual_shift: float
    certified_bound: float
    first_column_perturbation: float
    second_column_perturbation: float
    first_column_norm: float
    condition_number: float
    finite: bool


def r12_condition_number(block):
    a=np.asarray(block,dtype=np.float64); x=a[:,0]; y=a[:,1]; n=float(np.linalg.norm(x))
    if n<=1e-15: return float('inf')
    q=x/n; perp=y-q*float(q@y)
    return float(np.sqrt(1.0+float(perp@perp)/(n*n)))


def _bound_from_column_norms(base_block, first_column_perturbation, second_column_perturbation):
    base=np.asarray(base_block,dtype=np.float64)
    if base.shape!=(2,2): raise ValueError("implemented for 2x2 blocks")
    a,b=base[:,0],base[:,1]
    na=float(np.linalg.norm(a)); ne=float(first_column_perturbation); nf=float(second_column_perturbation)
    if na<=1e-15 or ne>=na:
        return float('inf')
    return float(nf + float(np.linalg.norm(b))*ne/(na-ne))


def finite_r12_perturbation_bound(base_block,perturbed_block):
    base=np.asarray(base_block,dtype=np.float64); pert=np.asarray(perturbed_block,dtype=np.float64)
    if base.shape!=(2,2) or pert.shape!=(2,2): raise ValueError("implemented for 2x2 blocks")
    a=base[:,0]; e=pert[:,0]-base[:,0]; f=pert[:,1]-base[:,1]
    na=float(np.linalg.norm(a)); ne=float(np.linalg.norm(e)); nf=float(np.linalg.norm(f))
    actual=abs(r12_value(pert)-r12_value(base))
    bound=_bound_from_column_norms(base,ne,nf)
    return R12PerturbationBound(float(actual),float(bound),ne,nf,na,r12_condition_number(base),bool(np.isfinite(bound)))


def group_pair_bound(base_blocks, perturbed_blocks, weights=None):
    """Deterministic bound for a single group perturbation.

    This is used for integer rounding and any other known base->perturbed pair.
    The returned theorem bound is a weighted sum of per-block finite R12 bounds.
    """
    if not base_blocks or len(base_blocks)!=len(perturbed_blocks):
        raise ValueError("base_blocks and perturbed_blocks must be non-empty and have equal length")
    w=unit_spread_weights(len(base_blocks)) if weights is None else np.asarray(weights,dtype=np.float64).ravel()
    w=w/np.linalg.norm(w)
    per=[finite_r12_perturbation_bound(a,b) for a,b in zip(base_blocks,perturbed_blocks)]
    bound=float(np.sum(np.abs(w)*np.asarray([x.certified_bound for x in per],dtype=np.float64)))
    base_stat=float(w@np.asarray([r12_value(b) for b in base_blocks]))
    pert_stat=float(w@np.asarray([r12_value(b) for b in perturbed_blocks]))
    return bound,abs(pert_stat-base_stat),tuple(float(x.certified_bound) for x in per)


def convex_hull_group_bound(base_blocks,extreme_block_sets,weights=None):
    """Certify the spread statistic for every convex mixture of extreme images.

    If an attacked block is A + sum_l alpha_l E_l, then the perturbation-column
    norms satisfy ||e|| <= max_l ||e_l|| and ||f|| <= max_l ||f_l||.  Applying
    the finite R12 inequality with these two maxima gives a conservative bound
    that is valid for *all* convex mixtures, even when different extremes
    maximize the two columns.  This is stronger/correcter than taking the max
    of already-combined per-extreme nonlinear bounds.
    """
    if not base_blocks: raise ValueError("base_blocks must be non-empty")
    w=unit_spread_weights(len(base_blocks)) if weights is None else np.asarray(weights,dtype=np.float64).ravel()
    w=w/np.linalg.norm(w)
    if not extreme_block_sets:
        return 0.0,0.0,[]
    for blocks in extreme_block_sets:
        if len(blocks)!=len(base_blocks): raise ValueError("each extreme set must match base_blocks")

    block_bounds=[]
    for i,base in enumerate(base_blocks):
        ne=[]; nf=[]
        for blocks in extreme_block_sets:
            e=np.asarray(blocks[i],dtype=np.float64)-np.asarray(base,dtype=np.float64)
            ne.append(float(np.linalg.norm(e[:,0])))
            nf.append(float(np.linalg.norm(e[:,1])))
        block_bounds.append(_bound_from_column_norms(base,max(ne),max(nf)))
    theorem=float(np.sum(np.abs(w)*np.asarray(block_bounds,dtype=np.float64)))

    base_stat=float(w@np.asarray([r12_value(b) for b in base_blocks]))
    observed=[]; extreme_pair_bounds=[]
    for blocks in extreme_block_sets:
        stat=float(w@np.asarray([r12_value(b) for b in blocks]))
        observed.append(abs(stat-base_stat))
        bnd,_,_=group_pair_bound(base_blocks,blocks,w)
        extreme_pair_bounds.append(float(bnd))
    return theorem,float(max(observed)),extreme_pair_bounds


def two_extreme_path_group_bound(base_blocks, extreme_block_sets, weights=None, *, subdivisions=8):
    """Rigorous bound specialized to a two-kernel convex convolution path.

    For two extreme attacked block sets ``B0`` and ``B1``, every point in the
    certified family is affine in the scalar mixture parameter ``alpha``:

        B(alpha) = (1-alpha) B0 + alpha B1,  alpha in [0,1].

    We partition this one-dimensional path.  On each interval, the attacked
    block is represented as a center block plus a perturbation whose two column
    norms are bounded exactly from the endpoint difference.  Applying the finite
    r12 perturbation inequality about that center and then triangle inequality
    yields a deterministic bound for the whole interval.  Taking the maximum
    over all intervals certifies the entire convex path.

    This theorem is often much tighter than the generic hull bound because it
    respects the fact that both attacked columns move with the *same* scalar
    mixture parameter instead of maximizing their perturbations independently.
    """
    if not base_blocks:
        raise ValueError("base_blocks must be non-empty")
    if len(extreme_block_sets) != 2:
        raise ValueError("two_extreme_path_group_bound requires exactly two extreme block sets")
    if int(subdivisions) < 1:
        raise ValueError("subdivisions must be positive")
    if any(len(blocks) != len(base_blocks) for blocks in extreme_block_sets):
        raise ValueError("each extreme set must match base_blocks")

    w = unit_spread_weights(len(base_blocks)) if weights is None else np.asarray(weights,dtype=np.float64).ravel()
    w = w / np.linalg.norm(w)
    b0 = [np.asarray(x,dtype=np.float64) for x in extreme_block_sets[0]]
    b1 = [np.asarray(x,dtype=np.float64) for x in extreme_block_sets[1]]
    base_stat = float(w @ np.asarray([r12_value(b) for b in base_blocks],dtype=np.float64))

    J = int(subdivisions)
    half_width = 0.5 / float(J)
    interval_bounds=[]
    for j in range(J):
        alpha = (float(j) + 0.5) / float(J)
        centers=[(1.0-alpha)*x0 + alpha*x1 for x0,x1 in zip(b0,b1)]
        center_stat=float(w @ np.asarray([r12_value(b) for b in centers],dtype=np.float64))
        local=[]
        for center,x0,x1 in zip(centers,b0,b1):
            diff=x1-x0
            ne=half_width*float(np.linalg.norm(diff[:,0]))
            nf=half_width*float(np.linalg.norm(diff[:,1]))
            local.append(_bound_from_column_norms(center,ne,nf))
        local_bound=float(np.sum(np.abs(w)*np.asarray(local,dtype=np.float64)))
        interval_bounds.append(abs(center_stat-base_stat)+local_bound)

    theorem=float(max(interval_bounds))
    observed=[]
    for blocks in extreme_block_sets:
        stat=float(w @ np.asarray([r12_value(b) for b in blocks],dtype=np.float64))
        observed.append(abs(stat-base_stat))
    return theorem,float(max(observed)),tuple(float(x) for x in interval_bounds)


def tightened_two_extreme_convex_group_bound(base_blocks, extreme_block_sets, weights=None, *, subdivisions=8):
    """Return the tighter of two independently valid convex-family bounds.

    The generic hull theorem remains the fallback.  Therefore enabling this
    helper can never weaken the previous certificate: the returned theorem bound
    is ``min(generic_bound, path_bound)`` and both terms are valid upper bounds.
    """
    generic, observed, generic_extreme = convex_hull_group_bound(base_blocks,extreme_block_sets,weights)
    if len(extreme_block_sets) != 2:
        return generic, observed, tuple(float(x) for x in generic_extreme), float('inf'), "generic", ()
    path, observed_path, intervals = two_extreme_path_group_bound(
        base_blocks,extreme_block_sets,weights,subdivisions=int(subdivisions)
    )
    if np.isfinite(path) and path < generic:
        return float(path),float(max(observed,observed_path)),tuple(float(x) for x in generic_extreme),float(generic),"two_extreme_piecewise_path",intervals
    return float(generic),float(max(observed,observed_path)),tuple(float(x) for x in generic_extreme),float(path),"generic",intervals

# ---------------------------------------------------------------------------
# Vectorized edge/reference kernels.  These are algebraically identical to the
# scalar helpers above, but operate on arrays shaped (groups, repetition, 2, 2).
# They are used by the real-time software reference and map directly to SIMD or
# an FPGA lane array.
# ---------------------------------------------------------------------------

def r12_values_batch(blocks):
    x=np.asarray(blocks,dtype=np.float64)
    if x.shape[-2:]!=(2,2):
        raise ValueError("r12_values_batch expects (...,2,2) blocks")
    first=x[..., :,0]; second=x[..., :,1]
    norm=np.sqrt(np.sum(first*first,axis=-1))
    dot=np.sum(first*second,axis=-1)
    out=np.empty_like(norm,dtype=np.float64)
    good=norm>1e-12
    out[good]=dot[good]/norm[good]
    if np.any(~good):
        flat_x=x.reshape((-1,2,2)); flat_out=out.reshape(-1); flat_good=good.reshape(-1)
        for idx in np.flatnonzero(~flat_good):
            flat_out[idx]=r12_value(flat_x[idx])
    return out


def _bound_from_column_norms_batch(base_blocks, first_column_perturbation, second_column_perturbation):
    base=np.asarray(base_blocks,dtype=np.float64)
    if base.shape[-2:]!=(2,2):
        raise ValueError("implemented for (...,2,2) blocks")
    ne=np.asarray(first_column_perturbation,dtype=np.float64)
    nf=np.asarray(second_column_perturbation,dtype=np.float64)
    a=base[..., :,0]; b=base[..., :,1]
    na=np.sqrt(np.sum(a*a,axis=-1)); nb=np.sqrt(np.sum(b*b,axis=-1))
    valid=(na>1e-15)&(ne<na)
    out=np.full(np.broadcast_shapes(na.shape,ne.shape,nf.shape),np.inf,dtype=np.float64)
    out[valid]=nf[valid]+nb[valid]*ne[valid]/(na[valid]-ne[valid])
    return out


def group_pair_bound_batch(base_blocks, perturbed_blocks, weights=None):
    base=np.asarray(base_blocks,dtype=np.float64); pert=np.asarray(perturbed_blocks,dtype=np.float64)
    if base.shape!=pert.shape or base.ndim!=4 or base.shape[-2:]!=(2,2):
        raise ValueError("batch group arrays must have shape (groups,repetition,2,2)")
    rep=base.shape[1]
    w=unit_spread_weights(rep) if weights is None else np.asarray(weights,dtype=np.float64).ravel()
    w=w/np.linalg.norm(w); aw=np.abs(w)[None,:]
    diff=pert-base
    ne=np.sqrt(np.sum(diff[..., :,0]**2,axis=-1)); nf=np.sqrt(np.sum(diff[..., :,1]**2,axis=-1))
    per=_bound_from_column_norms_batch(base,ne,nf)
    bound=np.sum(aw*per,axis=1)
    bstat=np.sum(r12_values_batch(base)*w[None,:],axis=1)
    pstat=np.sum(r12_values_batch(pert)*w[None,:],axis=1)
    return bound,np.abs(pstat-bstat),per


def convex_hull_group_bound_batch(base_blocks, extreme_block_sets, weights=None):
    base=np.asarray(base_blocks,dtype=np.float64); ext=np.asarray(extreme_block_sets,dtype=np.float64)
    if base.ndim!=4 or base.shape[-2:]!=(2,2):
        raise ValueError("base_blocks must have shape (groups,repetition,2,2)")
    if ext.ndim!=5 or ext.shape[1:]!=base.shape:
        raise ValueError("extremes must have shape (kernels,groups,repetition,2,2)")
    rep=base.shape[1]
    w=unit_spread_weights(rep) if weights is None else np.asarray(weights,dtype=np.float64).ravel()
    w=w/np.linalg.norm(w); aw=np.abs(w)[None,:]
    if ext.shape[0]==0:
        z=np.zeros(base.shape[0],dtype=np.float64)
        return z,z,np.empty((0,base.shape[0]),dtype=np.float64)
    diff=ext-base[None,...]
    ne=np.max(np.sqrt(np.sum(diff[..., :,0]**2,axis=-1)),axis=0)
    nf=np.max(np.sqrt(np.sum(diff[..., :,1]**2,axis=-1)),axis=0)
    per=_bound_from_column_norms_batch(base,ne,nf)
    theorem=np.sum(aw*per,axis=1)
    bstat=np.sum(r12_values_batch(base)*w[None,:],axis=1)
    estat=np.sum(r12_values_batch(ext)*w[None,None,:],axis=2)
    observed=np.max(np.abs(estat-bstat[None,:]),axis=0)
    pair=[]
    for k in range(ext.shape[0]):
        b,_,_=group_pair_bound_batch(base,ext[k],w); pair.append(b)
    return theorem,observed,np.stack(pair,axis=0)


def two_extreme_path_group_bound_batch(base_blocks, extreme_block_sets, weights=None, *, subdivisions=8):
    base=np.asarray(base_blocks,dtype=np.float64); ext=np.asarray(extreme_block_sets,dtype=np.float64)
    if ext.shape[0]!=2 or ext.shape[1:]!=base.shape:
        raise ValueError("two-extreme batch path requires shape (2,groups,repetition,2,2)")
    J=int(subdivisions)
    if J<1: raise ValueError("subdivisions must be positive")
    rep=base.shape[1]
    w=unit_spread_weights(rep) if weights is None else np.asarray(weights,dtype=np.float64).ravel()
    w=w/np.linalg.norm(w); aw=np.abs(w)[None,:]
    b0,b1=ext[0],ext[1]; diff=b1-b0
    bstat=np.sum(r12_values_batch(base)*w[None,:],axis=1)
    half=0.5/float(J); bounds=[]
    ne=half*np.sqrt(np.sum(diff[..., :,0]**2,axis=-1)); nf=half*np.sqrt(np.sum(diff[..., :,1]**2,axis=-1))
    for j in range(J):
        alpha=(float(j)+0.5)/float(J)
        center=(1.0-alpha)*b0+alpha*b1
        cstat=np.sum(r12_values_batch(center)*w[None,:],axis=1)
        local=_bound_from_column_norms_batch(center,ne,nf)
        bounds.append(np.abs(cstat-bstat)+np.sum(aw*local,axis=1))
    interval=np.stack(bounds,axis=0)
    theorem=np.max(interval,axis=0)
    estat=np.sum(r12_values_batch(ext)*w[None,None,:],axis=2)
    observed=np.max(np.abs(estat-bstat[None,:]),axis=0)
    return theorem,observed,interval


def tightened_two_extreme_convex_group_bound_batch(base_blocks, extreme_block_sets, weights=None, *, subdivisions=8):
    generic,observed,extreme=convex_hull_group_bound_batch(base_blocks,extreme_block_sets,weights)
    ext=np.asarray(extreme_block_sets,dtype=np.float64)
    if ext.shape[0]!=2:
        modes=np.zeros(generic.size,dtype=bool)
        return generic,observed,extreme,np.full_like(generic,np.inf),modes,np.empty((0,generic.size))
    path,observed_path,interval=two_extreme_path_group_bound_batch(base_blocks,extreme_block_sets,weights,subdivisions=subdivisions)
    use=np.isfinite(path)&(path<generic)
    chosen=np.where(use,path,generic)
    fallback=np.where(use,generic,path)
    return chosen,np.maximum(observed,observed_path),extreme,fallback,use,interval
