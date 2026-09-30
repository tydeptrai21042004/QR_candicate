from __future__ import annotations
import hashlib
import numpy as np
from ..core.config import ProposedConfig
from ..utils.permutation import selected_block_positions
from ..utils.watermark import arnold_transform,bits_from_watermark
from .certificate import choose_initial_spread_period_for_group,certify_spread_group
from .convolution import convolution_bank,reflect_convolve,spectral_distance_from_identity
from .side_info import build_side_info,side_info_overhead_bits
from .spread_qim import apply_spread_embedding,unit_spread_weights


def _blocks(channel,positions,bs):
    return [channel[r:r+bs,c:c+bs].astype(np.float64).copy() for r,c in positions]


def _rounded_candidates(cands):
    return [np.clip(np.rint(np.asarray(c,dtype=np.float64)),0,255).astype(np.float64) for c in cands]


def _group_sse(original_blocks,rounded_blocks):
    return float(sum(np.sum((np.asarray(a,dtype=np.float64)-np.asarray(b,dtype=np.float64))**2) for a,b in zip(original_blocks,rounded_blocks)))


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
    continuous_groups=[None]*n
    rounded_groups=[None]*n
    group_sses=np.zeros(n,dtype=np.float64)
    embedding_feasible=np.zeros(n,dtype=bool)
    range_feasible=np.zeros(n,dtype=bool)
    continuous_energy=np.zeros(n,dtype=np.float64)
    clipping_events=0

    def render_group(k:int,idx:int):
        gp=positions[k*cfg.repetition:(k+1)*cfg.repetition]
        base=_blocks(original_channel,gp,cfg.block_size)
        p=periods[int(idx)]
        cands,emb=apply_spread_embedding(base,int(bits[k]),p,weights)
        rounded=_rounded_candidates(cands)
        mse=float(emb.energy/(4.0*cfg.repetition)) if emb.feasible else float('inf')
        return gp,base,cands,rounded,emb,mse,_group_sse(base,rounded)

    # P1: initialize with the payload-independent lowest worst-bit-distortion
    # feasible period rather than maximizing Delta/certificate margin.
    for k,bit in enumerate(bits):
        gp=positions[k*cfg.repetition:(k+1)*cfg.repetition]
        base=_blocks(original_channel,gp,cfg.block_size)
        initial=choose_initial_spread_period_for_group(base,int(bit),cfg)
        idx=int(initial.period_index); period_indices[k]=idx
        gp,base,cands,rounded,emb,mse,sse=render_group(k,idx)
        if emb.feasible and mse<=float(cfg.max_group_mse):
            embedding_feasible[k]=True; range_feasible[k]=True
            continuous_groups[k]=cands; rounded_groups[k]=rounded
            continuous_energy[k]=float(emb.energy); group_sses[k]=sse
            for (r,c),cand,rb in zip(gp,cands,rounded):
                rr=np.rint(np.asarray(cand,dtype=np.float64))
                clipping_events+=int(np.any((rr<0)|(rr>255)))
                working[r:r+cfg.block_size,c:c+cfg.block_size]=rb
        else:
            # Explicit fail: no hidden fallback is presented as certified.
            embedding_feasible[k]=False; range_feasible[k]=False
            continuous_groups[k]=base; rounded_groups[k]=base
            group_sses[k]=0.0

    def certify_current():
        work=working[:h0,:w0]
        conv_channels=[reflect_convolve(work,k.kernel) for k in bank]
        decisions=[]
        for k in range(n):
            gp=positions[k*cfg.repetition:(k+1)*cfg.repetition]
            rounded=_blocks(work,gp,cfg.block_size)
            extremes=[_blocks(ch,gp,cfg.block_size) for ch in conv_channels]
            if not embedding_feasible[k]:
                # Still produce a diagnostic decision, but force uncertified.
                d=certify_spread_group(continuous_groups[k],rounded,extremes,int(bits[k]),int(period_indices[k]),cfg)
                d=d.__class__(**{**d.__dict__,"certified":False,"status":"embedding_infeasible"})
            else:
                d=certify_spread_group(continuous_groups[k],rounded,extremes,int(bits[k]),int(period_indices[k]),cfg)
            decisions.append(d)
        return decisions

    decisions=certify_current()
    passes_used=1

    # P0/P1: post-embedding adaptive escalation.  Only groups that are not
    # certified are moved to a larger feasible period.  After every batch the
    # *whole final rounded image* is convolved again and re-certified.
    for pass_no in range(1,int(cfg.certificate_max_passes)):
        uncert=[k for k,d in enumerate(decisions) if not d.certified and embedding_feasible[k]]
        if not uncert: break
        proposals=[]
        current_sse=float(group_sses.sum())
        for k in uncert:
            cur=int(period_indices[k]); best=None
            for idx in range(cur+1,len(periods)):
                gp,base,cands,rounded,emb,mse,sse=render_group(k,idx)
                if emb.feasible and mse<=float(cfg.max_group_mse):
                    best=(idx,gp,cands,rounded,emb,mse,sse); break
            if best is not None:
                delta=float(best[6]-group_sses[k]); proposals.append((delta,k,best))
        if not proposals: break
        changed=0
        # Spend the PSNR budget on the cheapest certification escalations first.
        for delta,k,best in sorted(proposals,key=lambda x:(x[0],x[1])):
            if delta>0 and current_sse+delta>sse_budget+1e-9:
                continue
            idx,gp,cands,rounded,emb,mse,sse=best
            period_indices[k]=idx; continuous_groups[k]=cands; rounded_groups[k]=rounded
            continuous_energy[k]=float(emb.energy); current_sse+=float(sse-group_sses[k]); group_sses[k]=sse
            for (r,c),rb in zip(gp,rounded): working[r:r+cfg.block_size,c:c+cfg.block_size]=rb
            changed+=1
        if changed==0: break
        decisions=certify_current(); passes_used=pass_no+1

    clipping_events=int(sum(np.any((np.rint(np.asarray(c))<0)|(np.rint(np.asarray(c))>255)) for group in continuous_groups for c in group))
    cert=np.asarray([bool(d.certified) for d in decisions],dtype=bool)
    bounds=np.asarray([float(d.theorem_bound) for d in decisions])
    round_bounds=np.asarray([float(d.rounding_bound) for d in decisions])
    residual=np.asarray([float(d.certificate_margin) for d in decisions])
    obs=np.asarray([float(d.convolution_shift) for d in decisions])
    obs_round=np.asarray([float(d.rounding_shift) for d in decisions])

    out=host.copy(); out[:h0,:w0,cfg.channel]=working[:h0,:w0].astype(np.uint8)
    actual_sse=float(np.sum((out.astype(np.float64)-host.astype(np.float64))**2))
    actual_psnr=float('inf') if actual_sse==0 else 10.0*np.log10(total_samples*255.0**2/actual_sse)
    hist={str(periods[i]):int(np.sum(period_indices==i)) for i in range(len(periods))}

    side_metadata={
        'method':'ccqr_r12_qim_v1',
        'algorithm_revision':'mc_ccqr_spread_qim_v3_postembed_cert',
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
        'algorithm_revision':'mc_ccqr_spread_qim_v3_postembed_cert',
        'certificate_reference_signal':'final_rounded_watermarked_image',
        'certified_family':'all convex mixtures of configured extreme reflect-101 convolution kernels',
        'certificate_condition':'rounding_bound + safety_factor*convolution_bound + additive_budget < period/4',
        'certified_fraction':float(cert.mean()),
        'certified_groups':int(cert.sum()),
        'uncertified_groups':int((~cert).sum()),
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
