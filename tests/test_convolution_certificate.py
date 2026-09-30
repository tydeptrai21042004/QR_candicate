import numpy as np
from qrwatermark.proposed.convolution import gaussian_kernel, circular_convolve, spectral_distance_from_identity
from qrwatermark.proposed.certificate import theoretical_group_mse


def test_circular_convolution_constant_fixed_point():
    x=np.full((32,32),17.0)
    k=gaussian_kernel(3,0.75)
    y=circular_convolve(x,k)
    assert np.max(np.abs(y-x)) < 1e-10


def test_spectral_identity_distance_positive_for_blur():
    k=gaussian_kernel(3,0.75)
    eta=spectral_distance_from_identity(k,(32,32))
    assert 0.0 < eta < 2.0


def test_exact_r12_distortion_model_nonnegative():
    values=np.asarray([10.0,20.0,30.0])
    d,ok=theoretical_group_mse(values,48.0)
    assert d >= 0.0 and ok


def test_post_embedding_certificate_survives_extremes_and_convex_mixtures():
    """Non-vacuous theorem regression on the exact final rounded watermark.

    The certified family is real-valued reflect-101 convolution.  The test checks
    both configured extreme kernels and unseen convex mixtures.  Integer
    requantization *after* the attack is intentionally outside this theorem.
    """
    from qrwatermark.core.config import ProposedConfig,NLMConfig
    from qrwatermark.proposed import ConvolutionCertifiedR12QIM
    from qrwatermark.proposed.convolution import convolution_bank,reflect_convolve
    from qrwatermark.proposed.side_info import unpack_certified_mask,unpack_period_indices
    from qrwatermark.proposed.soft_decoder import decode_groups
    from qrwatermark.utils.permutation import selected_block_positions
    from qrwatermark.utils.watermark import arnold_transform,bits_from_watermark

    h=w=128; yy,xx=np.mgrid[0:h,0:w]
    base=80.0+0.30*xx+0.20*yy
    host=np.stack([base,base+5.0,base+10.0],axis=2).clip(0,255).astype(np.uint8)
    rng=np.random.default_rng(7)
    wm=(rng.integers(0,2,size=(8,8),dtype=np.uint8)*255).astype(np.uint8)
    cfg=ProposedConfig(
        watermark_size=8,repetition=3,period_candidates=(24.0,28.0,32.0,36.0),
        convolution_gaussian_sigmas=(0.50,0.75),additive_feature_budget=0.0,
        target_psnr_db=45.0,certificate_max_passes=4,nlm=NLMConfig(enabled=False),
    )
    key=b'theorem-regression'
    emb=ConvolutionCertifiedR12QIM(cfg).embed(host,wm,key=key)
    certified=unpack_certified_mask(emb.side_info,key)
    assert int(certified.sum())>0, "the theorem regression must not pass vacuously"
    period_indices,meta=unpack_period_indices(emb.side_info,key)
    periods=np.asarray(meta['period_candidates'],dtype=np.float64)[period_indices]
    channel=emb.image[:,:,cfg.channel].astype(np.float64)
    h0=(h//cfg.block_size)*cfg.block_size; w0=(w//cfg.block_size)*cfg.block_size
    pos=selected_block_positions((h0,w0),cfg.block_size,len(periods)*cfg.repetition,key)
    truth=bits_from_watermark(arnold_transform(wm,cfg.arnold_iterations))
    bank=convolution_bank(cfg.convolution_kernel_size,cfg.convolution_gaussian_sigmas)
    kernels=[k.kernel for k in bank]
    kernels += [(1.0-a)*bank[0].kernel+a*bank[-1].kernel for a in (0.2,0.4,0.6,0.8)]
    for kernel in kernels:
        attacked=reflect_convolve(channel,kernel)
        decoded,_,_=decode_groups(attacked,pos,periods,cfg.repetition,cfg.block_size)
        assert np.array_equal(decoded[certified],truth[certified])


def test_deterministic_rounding_bound_dominates_observed_shift():
    from qrwatermark.proposed.qr_sensitivity import group_pair_bound
    from qrwatermark.proposed.spread_qim import apply_spread_embedding,unit_spread_weights
    rng=np.random.default_rng(9)
    blocks=[rng.uniform(40,210,size=(2,2)) for _ in range(5)]
    cands,emb=apply_spread_embedding(blocks,1,32.0,unit_spread_weights(5))
    assert emb.feasible
    rounded=[np.rint(x) for x in cands]
    theorem,observed,_=group_pair_bound(cands,rounded)
    assert np.isfinite(theorem)
    assert observed <= theorem + 1e-12
