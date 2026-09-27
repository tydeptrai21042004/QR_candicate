from __future__ import annotations
import hashlib
import numpy as np
from ..core.config import ProposedConfig
from ..utils.permutation import selected_block_positions
from ..utils.watermark import arnold_transform,bits_from_watermark
from .certificate import choose_spread_period_for_group
from .convolution import convolution_bank,reflect_convolve,spectral_distance_from_identity
from .side_info import build_side_info
from .spread_qim import apply_spread_embedding,unit_spread_weights

def _blocks(channel,positions,bs): return [channel[r:r+bs,c:c+bs].astype(np.float64).copy() for r,c in positions]

def embed_image(host,watermark,key,cfg:ProposedConfig):
    cfg.validate(); host=np.asarray(host,dtype=np.uint8); wm=np.asarray(watermark,dtype=np.uint8)
    if host.ndim!=3 or host.shape[2]<3: raise ValueError("host must be color")
    if wm.shape!=(cfg.watermark_size,cfg.watermark_size): raise ValueError(f"watermark must be {cfg.watermark_size}x{cfg.watermark_size}")
    bits=bits_from_watermark(arnold_transform(wm,cfg.arnold_iterations)); n=int(bits.size)
    channel=host[:,:,cfg.channel].astype(np.float64).copy(); h0=(channel.shape[0]//cfg.block_size)*cfg.block_size; w0=(channel.shape[1]//cfg.block_size)*cfg.block_size; work=channel[:h0,:w0]
    positions=selected_block_positions((h0,w0),cfg.block_size,n*cfg.repetition,key)
    bank=convolution_bank(cfg.convolution_kernel_size,cfg.convolution_gaussian_sigmas)
    conv_channels=[reflect_convolve(work,k.kernel) for k in bank]
    eta={k.name:spectral_distance_from_identity(k.kernel,work.shape) for k in bank}
    period_indices=np.zeros(n,dtype=np.uint8); decisions=[]
    for k in range(n):
        gp=positions[k*cfg.repetition:(k+1)*cfg.repetition]; base=_blocks(work,gp,cfg.block_size); extremes=[_blocks(ch,gp,cfg.block_size) for ch in conv_channels]
        d=choose_spread_period_for_group(base,extremes,cfg); period_indices[k]=d.period_index; decisions.append(d)
    weights=unit_spread_weights(cfg.repetition); clip=0; infeasible=0; energy=0.0
    for k,bit in enumerate(bits):
        p=float(cfg.period_candidates[int(period_indices[k])]); gp=positions[k*cfg.repetition:(k+1)*cfg.repetition]; base=_blocks(channel,gp,cfg.block_size); cands,emb=apply_spread_embedding(base,int(bit),p,weights)
        if not emb.feasible: infeasible+=1; continue
        energy+=emb.energy
        for (r,c),cand in zip(gp,cands):
            rounded=np.rint(cand); clip+=int(np.any((rounded<0)|(rounded>255))); channel[r:r+cfg.block_size,c:c+cfg.block_size]=np.clip(rounded,0,255)
    out=host.copy(); out[:h0,:w0,cfg.channel]=channel[:h0,:w0].astype(np.uint8)
    periods=[float(x) for x in cfg.period_candidates]
    metadata={'method':'ccqr_r12_qim_v1','algorithm_revision':'mc_ccqr_spread_qim_v2','watermark_shape':[cfg.watermark_size,cfg.watermark_size],'repetition':cfg.repetition,'spread_weights':'equal_unit_norm','block_size':cfg.block_size,'channel':cfg.channel,'arnold_iterations':cfg.arnold_iterations,'period_candidates':periods,'period_count':len(periods),'target_psnr_db':float(cfg.target_psnr_db),'convolution_boundary':'reflect_101','convolution_uncertainty':'convex_hull_of_extreme_kernels','convolution_kernel_size':cfg.convolution_kernel_size,'convolution_gaussian_sigmas':[float(x) for x in cfg.convolution_gaussian_sigmas],'convolution_spectral_eta_circular_diagnostic':eta,'key_fingerprint':hashlib.sha256(key).hexdigest()[:16]}
    side=build_side_info(period_indices,metadata,key)
    cert=np.asarray([d.certified for d in decisions]); bounds=np.asarray([d.theorem_bound for d in decisions]); residual=np.asarray([d.certificate_margin for d in decisions]); obs=np.asarray([d.convolution_shift for d in decisions]); mse=np.asarray([d.worst_case_mse for d in decisions]); rangeok=np.asarray([d.range_feasible for d in decisions])
    hist={str(periods[i]):int(np.sum(period_indices==i)) for i in range(len(periods))}
    total=float(host.shape[0]*host.shape[1]*3); sse=float(np.sum((out.astype(np.float64)-host.astype(np.float64))**2)); budget=total*255.0**2*10.0**(-cfg.target_psnr_db/10.0); psnr=float('inf') if sse==0 else 10.0*np.log10(total*255.0**2/sse)
    finite=bounds[np.isfinite(bounds)]
    meta={'method':'ccqr_r12_qim_v1','algorithm_revision':'mc_ccqr_spread_qim_v2','certified_family':'all convex mixtures of configured extreme reflect-convolution kernels','certified_fraction':float(cert.mean()),'range_feasible_fraction':float(rangeok.mean()),'mean_certificate_margin':float(residual.mean()),'min_certificate_margin':float(residual.min()),'mean_observed_extreme_statistic_shift':float(obs.mean()),'max_observed_extreme_statistic_shift':float(obs.max()),'mean_finite_qr_theorem_bound':float(finite.mean()) if finite.size else float('inf'),'max_finite_qr_theorem_bound':float(finite.max()) if finite.size else float('inf'),'finite_bound_fraction':float(np.isfinite(bounds).mean()),'mean_worst_hypothetical_group_mse':float(mse.mean()),'period_histogram':hist,'clipping_events':int(clip),'infeasible_groups':int(infeasible),'continuous_embedding_energy':float(energy),'target_psnr_db':float(cfg.target_psnr_db),'target_sse_budget':float(budget),'actual_rgb_sse':sse,'actual_psnr_db':psnr,'target_psnr_met':bool(sse<=budget+1e-9),'convolution_spectral_eta_circular_diagnostic':eta}
    return out,side,meta
