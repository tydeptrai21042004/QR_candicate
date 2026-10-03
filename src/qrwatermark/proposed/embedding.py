from __future__ import annotations
import hashlib
import numpy as np
from ..core.config import ProposedConfig
from ..utils.permutation import selected_block_positions
from ..utils.watermark import arnold_transform,bits_from_watermark
from .certificate import choose_initial_spread_period_for_group,certify_spread_group,certify_spread_groups_batch,certify_spread_groups_arrays
from .convolution import convolution_bank,reflect_convolve,spectral_distance_from_identity
from .side_info import build_side_info,side_info_overhead_bits
from .spread_qim import apply_spread_embedding,unit_spread_weights,minimum_energy_box_projection_batch


def _blocks(channel,positions,bs):
    return [channel[r:r+bs,c:c+bs].astype(np.float64).copy() for r,c in positions]


def _rounded_candidates(cands):
    return [np.clip(np.rint(np.asarray(c,dtype=np.float64)),0,255).astype(np.float64) for c in cands]


def _group_sse(original_blocks,rounded_blocks):
    return float(sum(np.sum((np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))**2) for a,b in zip(original_blocks,rounded_blocks)))


def _prepare_candidate_bank(base_groups,bits,periods,weights,max_group_mse):
    """Precompute every period candidate using the exact spread-QIM equations.

    This removes period/group Python loops from the frame datapath.  The common
    unconstrained KKT solution is fully vectorized; rare box-active groups call
    the original scalar solver, so candidate semantics remain unchanged.
    """
    base=np.asarray(base_groups,dtype=np.float64)
    bits=np.asarray(bits,dtype=np.uint8).ravel(); periods=np.asarray(periods,dtype=np.float64)
    n,rep=base.shape[:2]; w=np.asarray(weights,dtype=np.float64); w=w/np.linalg.norm(w)
    first=base[..., :,0]; second=base[..., :,1]
    norm=np.sqrt(np.sum(first*first,axis=-1)); q1=np.zeros_like(first)
    good=norm>1e-12; q1[good]=first[good]/norm[good,None]
    # Degenerate first columns retain the scalar canonical-QR geometry.
    if np.any(~good):
        from .qr import canonical_qr
        for gi,ri in np.argwhere(~good):
            q,_=canonical_qr(base[gi,ri]); q1[gi,ri]=q[:,0]
    values=np.sum(first*second,axis=-1)/np.where(good,norm,1.0)
    if np.any(~good):
        from .r_branch import r12_value
        for gi,ri in np.argwhere(~good): values[gi,ri]=r12_value(base[gi,ri])
    stat=np.sum(values*w[None,:],axis=1)

    # Exact per-carrier delta interval keeping the reconstructed second column
    # in [0,255]. Original pixels are uint8, so qi==0 contributes no bound.
    lo=np.full((n,rep),-np.inf); hi=np.full((n,rep),np.inf)
    for row in range(2):
        qi=q1[...,row]; xi=second[...,row]; nz=np.abs(qi)>1e-15
        a0=np.zeros_like(qi); a1=np.zeros_like(qi)
        a0[nz]=(0.0-xi[nz])/qi[nz]; a1[nz]=(255.0-xi[nz])/qi[nz]
        lower=np.minimum(a0,a1); upper=np.maximum(a0,a1)
        lo=np.maximum(lo,np.where(nz,lower,-np.inf)); hi=np.minimum(hi,np.where(nz,upper,np.inf))

    P=periods.size
    bit_energy=np.full((2,P,n),np.inf); bit_ok=np.zeros((2,P,n),dtype=bool)
    actual_delta=np.zeros((P,n,rep),dtype=np.float64); actual_ok=np.zeros((P,n),dtype=bool)
    for pi,p in enumerate(periods):
        for bit in (0,1):
            offset=(0.25 if bit==0 else 0.75)*p
            target=offset+np.rint((stat-offset)/p)*p
            d=target-stat
            delta,ok=minimum_energy_box_projection_batch(d,w,lo,hi)
            energy=np.sum(delta*delta,axis=1)
            bit_energy[bit,pi]=np.where(ok,energy,np.inf); bit_ok[bit,pi]=ok
            take=(bits==bit)
            actual_delta[pi,take]=delta[take]; actual_ok[pi,take]=ok[take]
    worst_mse=np.max(bit_energy,axis=0)/(4.0*float(rep))
    range_ok=np.all(bit_ok,axis=0)

    continuous=np.repeat(base[None,...],P,axis=0)
    continuous[..., :,1]+=actual_delta[...,None]*q1[None,...]
    # Preserve the legacy NumPy-QR rounding result at the vanishingly rare
    # exact half-integer boundary, matching apply_r12_delta's compatibility
    # guard while keeping every normal carrier on the vectorized path.
    frac=continuous[..., :,1]-np.floor(continuous[..., :,1])
    near_half=np.any(np.abs(frac-0.5)<=1e-10,axis=-1)
    if np.any(near_half):
        from .r_branch import apply_r12_delta
        for pi,gi,ri in np.argwhere(near_half):
            continuous[pi,gi,ri]=apply_r12_delta(base[gi,ri],float(actual_delta[pi,gi,ri]))
    rounded=np.clip(np.rint(continuous),0,255).astype(np.float64)
    actual_energy=np.sum(actual_delta*actual_delta,axis=2)
    actual_mse=actual_energy/(4.0*float(rep))
    actual_sse=np.sum((rounded-base[None,...])**2,axis=(2,3,4))
    actual_ok &= actual_mse<=float(max_group_mse)
    return {
        'continuous':continuous,'rounded':rounded,'energy':actual_energy,
        'mse':actual_mse,'sse':actual_sse,'actual_ok':actual_ok,
        'worst_mse':worst_mse,'range_ok':range_ok,
    }


def embed_image(host,watermark,key,cfg:ProposedConfig):
    cfg.validate(); host=np.asarray(host,dtype=np.uint8); wm=np.asarray(watermark,dtype=np.uint8)
    if host.ndim!=3 or host.shape[2]<3: raise ValueError("host must be color")
    if wm.shape!=(cfg.watermark_size,cfg.watermark_size): raise ValueError(f"watermark must be {cfg.watermark_size}x{cfg.watermark_size}")

    bits=bits_from_watermark(arnold_transform(wm,cfg.arnold_iterations)); n=int(bits.size)
    original_channel=host[:,:,cfg.channel].astype(np.float64).copy()
    h0=(original_channel.shape[0]//cfg.block_size)*cfg.block_size
    w0=(original_channel.shape[1]//cfg.block_size)*cfg.block_size
    original_work=original_channel[:h0,:w0]
    positions=selected_block_positions((h0,w0),cfg.block_size,n*cfg.repetition,key)
    weights=unit_spread_weights(cfg.repetition)
    bank=convolution_bank(cfg.convolution_kernel_size,cfg.convolution_gaussian_sigmas)
    eta={k.name:spectral_distance_from_identity(k.kernel,original_work.shape) for k in bank}
    periods=[float(x) for x in cfg.period_candidates]

    # Global RGB PSNR budget.  Only one channel is modified, but the denominator
    # remains all RGB samples so the reported target is the true RGB PSNR.
    total_samples=float(host.shape[0]*host.shape[1]*3)
    sse_budget=total_samples*255.0**2*10.0**(-float(cfg.target_psnr_db)/10.0)

    period_indices=np.zeros(n,dtype=np.uint8)
    working=original_channel.copy()
    embedding_feasible=np.zeros(n,dtype=bool)
    range_feasible=np.zeros(n,dtype=bool)
    continuous_energy=np.zeros(n,dtype=np.float64)
    clipping_events=0

    # Gather the immutable original selected blocks once and precompute all four
    # actual-bit candidates plus both-bit worst-case distortion decisions.
    _pos0=np.asarray(positions,dtype=np.int64); _r0=_pos0[:,0]; _c0=_pos0[:,1]
    _flat=np.empty((_r0.size,2,2),dtype=np.float64)
    _flat[:,0,0]=original_channel[_r0,_c0]; _flat[:,0,1]=original_channel[_r0,_c0+1]
    _flat[:,1,0]=original_channel[_r0+1,_c0]; _flat[:,1,1]=original_channel[_r0+1,_c0+1]
    base_groups=_flat.reshape(n,cfg.repetition,2,2)
    bank_candidates=_prepare_candidate_bank(base_groups,bits,periods,weights,cfg.max_group_mse)

    # Exact initial selector: same lexicographic rule as
    # choose_initial_spread_period_for_group, evaluated in a batch.
    worst=bank_candidates['worst_mse']; rok=bank_candidates['range_ok']
    dist=worst<=float(cfg.max_group_mse); feasible=dist&rok
    for k in range(n):
        inds=np.flatnonzero(feasible[:,k])
        if inds.size:
            idx=min((int(i) for i in inds),key=lambda i:(float(worst[i,k]),periods[i]))
        else:
            idx=min(range(len(periods)),key=lambda i:(not bool(rok[i,k]),float(worst[i,k]),periods[i]))
        period_indices[k]=idx

    gidx=np.arange(n); chosen=period_indices.astype(np.int64)
    embedding_feasible=bank_candidates['actual_ok'][chosen,gidx].copy()
    range_feasible=embedding_feasible.copy()
    continuous_groups=bank_candidates['continuous'][chosen,gidx].copy()
    rounded_groups=bank_candidates['rounded'][chosen,gidx].copy()
    continuous_energy=np.where(embedding_feasible,bank_candidates['energy'][chosen,gidx],0.0)
    group_sses=np.where(embedding_feasible,bank_candidates['sse'][chosen,gidx],0.0)
    # Explicit fail groups remain unchanged.
    continuous_groups[~embedding_feasible]=base_groups[~embedding_feasible]
    rounded_groups[~embedding_feasible]=base_groups[~embedding_feasible]

    # Scatter selected rounded blocks back to the channel. Positions are unique.
    flat_round=rounded_groups.reshape(-1,2,2); flat_ok=np.repeat(embedding_feasible,cfg.repetition)
    rr=_r0[flat_ok]; cc=_c0[flat_ok]; rb=flat_round[flat_ok]
    working[rr,cc]=rb[:,0,0]; working[rr,cc+1]=rb[:,0,1]
    working[rr+1,cc]=rb[:,1,0]; working[rr+1,cc+1]=rb[:,1,1]

    def render_group(k:int,idx:int):
        gp=positions[k*cfg.repetition:(k+1)*cfg.repetition]
        base=[b.copy() for b in base_groups[k]]
        cands=[b.copy() for b in bank_candidates['continuous'][idx,k]]
        rounded=[b.copy() for b in bank_candidates['rounded'][idx,k]]
        class _Emb:
            pass
        emb=_Emb(); emb.feasible=bool(bank_candidates['actual_ok'][idx,k]); emb.energy=float(bank_candidates['energy'][idx,k])
        mse=float(bank_candidates['mse'][idx,k]); sse=float(bank_candidates['sse'][idx,k])
        return gp,base,cands,rounded,emb,mse,sse

    # Position arrays are session-invariant and reused by the vectorized
    # certification path.
    _pos=np.asarray(positions,dtype=np.int64)
    _rr=_pos[:,0]; _cc=_pos[:,1]

    def _gather_selected_blocks(channel):
        x=np.asarray(channel,dtype=np.float64)
        out=np.empty((_rr.size,2,2),dtype=np.float64)
        out[:,0,0]=x[_rr,_cc]; out[:,0,1]=x[_rr,_cc+1]
        out[:,1,0]=x[_rr+1,_cc]; out[:,1,1]=x[_rr+1,_cc+1]
        return out.reshape(n,cfg.repetition,2,2)

    _cert_cache={}
    def certify_current(*,tighten_final=False,reuse_current=False):
        if reuse_current and _cert_cache:
            rounded=_cert_cache['rounded']; extremes=_cert_cache['extremes']
        else:
            work=working[:h0,:w0]
            conv_channels=[reflect_convolve(work,k.kernel) for k in bank]
            rounded=_gather_selected_blocks(work)
            # An empty bank is a valid ablation: it removes the convolution-family
            # term while retaining deterministic rounding/additive-budget checks.
            # Keep the kernel axis explicitly empty so the vectorized certificate
            # can return a zero convolution bound instead of np.stack() failing.
            if conv_channels:
                extremes=np.stack([_gather_selected_blocks(ch) for ch in conv_channels],axis=0)
            else:
                extremes=np.empty((0,n,cfg.repetition,2,2),dtype=np.float64)
            _cert_cache['rounded']=rounded; _cert_cache['extremes']=extremes
        continuous=np.asarray(continuous_groups,dtype=np.float64)
        return certify_spread_groups_arrays(
            continuous,rounded,extremes,period_indices,cfg,
            tighten_final=tighten_final,embedding_feasible=embedding_feasible
        )

    decisions=certify_current()
    passes_used=1

    # P0/P1: post-embedding adaptive escalation.  Only groups that are not
    # certified are moved to a larger feasible period.  After every batch the
    # *whole final rounded image* is convolved again and re-certified.
    for pass_no in range(1,int(cfg.certificate_max_passes)):
        cert_now=np.asarray(decisions['certified'],dtype=bool)
        uncert=np.flatnonzero((~cert_now)&embedding_feasible)
        if uncert.size==0: break

        # Find the first larger feasible period for every uncertified group.
        next_idx=np.full(n,-1,dtype=np.int16); cur=period_indices.astype(np.int16)
        for idx in range(len(periods)):
            mask=(next_idx<0)&(idx>cur)&bank_candidates['actual_ok'][idx]
            next_idx[mask]=idx
        cand_groups=uncert[next_idx[uncert]>=0]
        if cand_groups.size==0: break
        cand_periods=next_idx[cand_groups].astype(np.int64)
        new_sse=bank_candidates['sse'][cand_periods,cand_groups]
        delta=new_sse-group_sses[cand_groups]
        order=np.lexsort((cand_groups,delta))
        cg=cand_groups[order]; cp=cand_periods[order]; dd=delta[order]
        current_sse=float(group_sses.sum())

        # The legacy loop processes negative deltas and then the cheapest positive
        # prefix until the fixed PSNR budget is exhausted.  Because dd is sorted,
        # this prefix can be selected exactly with a cumulative sum.
        csum=np.cumsum(dd); within=(current_sse+csum)<=sse_budget+1e-9
        if np.any(~within):
            first_fail=int(np.flatnonzero(~within)[0]); take=np.arange(dd.size)<first_fail
        else:
            take=np.ones(dd.size,dtype=bool)
        # Negative deltas are always accepted by the scalar reference; sorted
        # order guarantees they occur before any positive budget failure.
        sel=np.flatnonzero(take)
        if sel.size==0: break
        groups=cg[sel]; pidx=cp[sel]

        period_indices[groups]=pidx.astype(np.uint8)
        continuous_groups[groups]=bank_candidates['continuous'][pidx,groups]
        rounded_groups[groups]=bank_candidates['rounded'][pidx,groups]
        continuous_energy[groups]=bank_candidates['energy'][pidx,groups]
        group_sses[groups]=bank_candidates['sse'][pidx,groups]

        # Scatter all changed carriers in one batch.
        carrier_idx=(groups[:,None]*cfg.repetition+np.arange(cfg.repetition)[None,:]).reshape(-1)
        rb=rounded_groups[groups].reshape(-1,2,2); rr=_r0[carrier_idx]; cc=_c0[carrier_idx]
        working[rr,cc]=rb[:,0,0]; working[rr,cc+1]=rb[:,0,1]
        working[rr+1,cc]=rb[:,1,0]; working[rr+1,cc+1]=rb[:,1,1]

        decisions=certify_current(); passes_used=pass_no+1

    # The adaptive embedding above intentionally keeps the original conservative
    # generic certificate.  Only after the final pixels and periods are frozen do
    # we optionally apply the tighter two-extreme path theorem.  Consequently
    # this v4 certification pass can change only the certificate mask/metadata --
    # never the watermarked image, PSNR, payload, or period allocation.
    generic_decisions=decisions
    if bool(cfg.certificate_final_tighten) and len(bank)==2:
        decisions=certify_current(tighten_final=True,reuse_current=True)

    _rrnd=np.rint(np.asarray(continuous_groups,dtype=np.float64))
    clipping_events=int(np.sum(np.any((_rrnd<0)|(_rrnd>255),axis=(2,3))))
    generic_cert=np.asarray(generic_decisions['certified'],dtype=bool)
    cert=np.asarray(decisions['certified'],dtype=bool)
    bounds=np.asarray(decisions['convolution_bound'],dtype=np.float64)
    round_bounds=np.asarray(decisions['rounding_bound'],dtype=np.float64)
    residual=np.asarray(decisions['certificate_margin'],dtype=np.float64)
    obs=np.asarray(decisions['convolution_shift'],dtype=np.float64)
    obs_round=np.asarray(decisions['rounding_shift'],dtype=np.float64)

    out=host.copy(); out[:h0,:w0,cfg.channel]=working[:h0,:w0].astype(np.uint8)
    actual_sse=float(np.sum((out.astype(np.float64)-host.astype(np.float64))**2))
    actual_psnr=float('inf') if actual_sse==0 else 10.0*np.log10(total_samples*255.0**2/actual_sse)
    hist={str(periods[i]):int(np.sum(period_indices==i)) for i in range(len(periods))}

    side_metadata={
        'method':'ccqr_r12_qim_v1',
        'algorithm_revision':'mc_ccqr_spread_qim_v4_hw_path_cert',
        'watermark_shape':[cfg.watermark_size,cfg.watermark_size],
        'repetition':cfg.repetition,
        'spread_weights':'equal_unit_norm',
        'block_size':cfg.block_size,
        'channel':cfg.channel,
        'arnold_iterations':cfg.arnold_iterations,
        'period_candidates':periods,
        'period_count':len(periods),
        'target_psnr_db':float(cfg.target_psnr_db),
        'convolution_boundary':'reflect_101',
        'convolution_uncertainty':'convex_hull_of_extreme_kernels',
        'convolution_kernel_size':cfg.convolution_kernel_size,
        'convolution_gaussian_sigmas':[float(x) for x in cfg.convolution_gaussian_sigmas],
        'rounding_certificate':'deterministic_finite_r12_bound_continuous_to_uint8',
        'final_certificate_bound':'piecewise_two_extreme_convex_path_with_generic_fallback' if bool(cfg.certificate_final_tighten) and len(bank)==2 else 'generic_convex_hull',
        'certificate_path_subdivisions':int(cfg.certificate_path_subdivisions),
        'key_fingerprint':hashlib.sha256(key).hexdigest()[:16],
    }
    side=build_side_info(period_indices,side_metadata,key,certified_mask=cert)
    overhead=side_info_overhead_bits(side)

    def finite_stats(arr):
        a=np.asarray(arr,dtype=np.float64); f=a[np.isfinite(a)]
        return (float(f.mean()),float(f.min()),float(f.max()),float(f.size/max(1,a.size))) if f.size else (float('inf'),float('-inf'),float('inf'),0.0)
    mean_bound,min_bound,max_bound,finite_bound_fraction=finite_stats(bounds)
    mean_rbound,min_rbound,max_rbound,finite_round_fraction=finite_stats(round_bounds)
    finite_margin=residual[np.isfinite(residual)]

    meta={
        'method':'ccqr_r12_qim_v1',
        'algorithm_revision':'mc_ccqr_spread_qim_v4_hw_path_cert',
        'certificate_reference_signal':'final_rounded_watermarked_image',
        'certified_family':'all convex mixtures of configured extreme reflect-101 convolution kernels',
        'certificate_condition':'rounding_bound + safety_factor*convolution_bound + additive_budget < period/4',
        'certified_fraction':float(cert.mean()),
        'certified_groups':int(cert.sum()),
        'uncertified_groups':int((~cert).sum()),
        'generic_certificate_fraction_before_tightening':float(generic_cert.mean()),
        'generic_certified_groups_before_tightening':int(generic_cert.sum()),
        'newly_certified_by_path_bound':int(np.sum(cert & ~generic_cert)),
        'final_certificate_bound_modes':{'two_extreme_piecewise_path':int(np.sum(decisions['use_path'])),'generic':int(n-np.sum(decisions['use_path']))},
        'certificate_path_subdivisions':int(cfg.certificate_path_subdivisions),
        'embedding_infeasible_groups':int((~embedding_feasible).sum()),
        'certificate_passes_used':int(passes_used),
        'mean_certificate_margin':float(finite_margin.mean()) if finite_margin.size else float('-inf'),
        'min_certificate_margin':float(finite_margin.min()) if finite_margin.size else float('-inf'),
        'mean_observed_extreme_statistic_shift':float(obs.mean()),
        'max_observed_extreme_statistic_shift':float(obs.max()),
        'mean_observed_rounding_statistic_shift':float(obs_round.mean()),
        'max_observed_rounding_statistic_shift':float(obs_round.max()),
        'mean_finite_convolution_theorem_bound':mean_bound,
        'max_finite_convolution_theorem_bound':max_bound,
        'finite_convolution_bound_fraction':finite_bound_fraction,
        'mean_finite_rounding_theorem_bound':mean_rbound,
        'max_finite_rounding_theorem_bound':max_rbound,
        'finite_rounding_bound_fraction':finite_round_fraction,
        'period_histogram':hist,
        'period_selection':'payload_independent_lowest_worst_bit_distortion_then_post_embedding_escalation',
        'clipping_events':int(clipping_events),
        'continuous_embedding_energy':float(continuous_energy.sum()),
        'target_psnr_db':float(cfg.target_psnr_db),
        'target_sse_budget':float(sse_budget),
        'actual_rgb_sse':actual_sse,
        'actual_psnr_db':actual_psnr,
        'target_psnr_met':bool(actual_sse<=sse_budget+1e-9),
        'convolution_spectral_eta_circular_diagnostic':eta,
        'side_information_overhead_bits':overhead,
    }
    return out,side,meta
