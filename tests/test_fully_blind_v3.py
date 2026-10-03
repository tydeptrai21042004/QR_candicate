from __future__ import annotations

import numpy as np

from qrwatermark.core.config import ProposedConfig
from qrwatermark.evaluation.metrics import ber
from qrwatermark.proposed import ConvolutionCertifiedR12QIM
from qrwatermark.utils.permutation import selected_block_positions


def _cfg(**kw):
    base = dict(
        design="blind_v3",
        watermark_size=8,
        qim_period=48.0,
        blind_margin_ratio=0.125,
        blind_projection="safe_set",
        compute_certificate=False,
        store_certificate_mask=False,
    )
    base.update(kw)
    return ProposedConfig(**base)


def _sample(seed=301):
    rng = np.random.default_rng(seed)
    # Keep the synthetic host away from saturation so the fixed carrier
    # projection is feasible without needing any content-dependent skip map.
    host = rng.integers(24, 232, size=(64, 64, 3), dtype=np.uint8)
    wm = (rng.integers(0, 2, size=(8, 8), dtype=np.uint8) * 255).astype(np.uint8)
    return host, wm


def test_blind_roundtrip_has_literal_zero_side_information():
    host, wm = _sample()
    method = ConvolutionCertifiedR12QIM(_cfg())
    emb = method.embed(host, wm, key=b"blind-roundtrip")
    assert emb.side_info is None
    assert emb.metadata["fully_blind"] is True
    assert emb.metadata["requires_side_information"] is False
    assert emb.metadata["side_information_bits"] == 0
    assert emb.metadata["side_information_overhead_bits"]["total_serialized_bits"] == 0

    ext = method.extract(
        emb.image,
        key=b"blind-roundtrip",
        side_info=None,
        watermark_shape=wm.shape,
    )
    assert ext.metadata["side_information_used"] is False
    assert ext.metadata["original_host_used"] is False
    assert ber(wm, ext.watermark) == 0.0


def test_only_key_determined_carriers_can_change():
    host, wm = _sample(302)
    key = b"key-only-map"
    method = ConvolutionCertifiedR12QIM(_cfg())
    emb = method.embed(host, wm, key=key)

    positions = selected_block_positions((64, 64), 2, wm.size, key)
    allowed = np.zeros(host.shape[:2], dtype=bool)
    # v3 changes only the second QR column; the first image column of every
    # selected 2x2 block remains untouched.
    for r, c in positions:
        allowed[r:r+2, c+1] = True
    changed = np.any(emb.image != host, axis=2)
    assert not np.any(changed & ~allowed)


def test_safe_set_projection_is_no_worse_than_center_qim_energy():
    host, wm = _sample(303)
    key = b"projection-ablation"
    safe = ConvolutionCertifiedR12QIM(_cfg(blind_projection="safe_set"))
    center = ConvolutionCertifiedR12QIM(_cfg(blind_projection="center"))
    es = safe.embed(host, wm, key=key)
    ec = center.embed(host, wm, key=key)
    assert es.metadata["continuous_embedding_energy"] <= ec.metadata["continuous_embedding_energy"] + 1e-9
    xs = safe.extract(es.image, key=key, side_info=None, watermark_shape=wm.shape)
    xc = center.extract(ec.image, key=key, side_info=None, watermark_shape=wm.shape)
    assert ber(wm, xs.watermark) == 0.0
    assert ber(wm, xc.watermark) == 0.0


def test_certificate_is_reporting_only_and_emits_no_mask():
    host, wm = _sample(304)
    key = b"blind-cert-independent"
    a = ConvolutionCertifiedR12QIM(
        _cfg(compute_certificate=True, convolution_gaussian_sigmas=(0.5, 0.75))
    )
    b = ConvolutionCertifiedR12QIM(_cfg(compute_certificate=False))
    ea = a.embed(host, wm, key=key)
    eb = b.embed(host, wm, key=key)
    assert ea.side_info is None and eb.side_info is None
    assert np.array_equal(ea.image, eb.image)


def test_wrong_key_does_not_recover_payload():
    host, wm = _sample(305)
    method = ConvolutionCertifiedR12QIM(_cfg())
    emb = method.embed(host, wm, key=b"right-key")
    wrong = method.extract(emb.image, key=b"wrong-key", side_info=None, watermark_shape=wm.shape)
    assert ber(wm, wrong.watermark) > 0.20
