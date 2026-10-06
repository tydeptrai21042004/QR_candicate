from __future__ import annotations

import numpy as np

from qrwatermark.core.config import ProposedConfig
from qrwatermark.evaluation.metrics import ber
from qrwatermark.proposed import ConvolutionCertifiedR12QIM
from qrwatermark.utils.permutation import selected_block_positions


def _cfg(**kw):
    base = dict(
        design="blind_v4",
        watermark_size=8,
        coupled_q_period=0.20,
        coupled_r_period=0.48,
        coupled_q_margin_ratio=0.08,
        coupled_r_margin_ratio=0.08,
        coupled_q_weight=0.5,
        coupled_r_weight=1.0,
        coupled_energy_tau=16.0,
        compute_certificate=False,
        store_certificate_mask=False,
    )
    base.update(kw)
    return ProposedConfig(**base)


def _sample(seed=901):
    rng = np.random.default_rng(seed)
    host = rng.integers(16, 240, size=(64, 64, 3), dtype=np.uint8)
    wm = (rng.integers(0, 2, size=(8, 8), dtype=np.uint8) * 255).astype(np.uint8)
    return host, wm


def test_v4_is_literal_zero_side_information_and_clean_roundtrip():
    host, wm = _sample()
    method = ConvolutionCertifiedR12QIM(_cfg())
    emb = method.embed(host, wm, key=b"v4-roundtrip")
    assert method.name == "blind_cqr_qim_v4"
    assert emb.side_info is None
    assert emb.metadata["fully_blind"] is True
    assert emb.metadata["requires_side_information"] is False
    assert emb.metadata["side_information_bits"] == 0
    assert emb.metadata["repetition_across_blocks"] is False
    assert emb.metadata["payload_embeddings_per_bit"] == 1
    ext = method.extract(emb.image, key=b"v4-roundtrip", side_info=None, watermark_shape=wm.shape)
    assert ext.metadata["side_information_used"] is False
    assert ext.metadata["original_host_used"] is False
    assert ber(wm, ext.watermark) == 0.0


def test_v4_changes_only_one_keyed_2x2_block_per_payload_bit():
    host, wm = _sample(902)
    key = b"v4-one-block"
    method = ConvolutionCertifiedR12QIM(_cfg())
    emb = method.embed(host, wm, key=key)
    positions = selected_block_positions((64, 64), 2, wm.size, key)
    allowed = np.zeros(host.shape[:2], dtype=bool)
    for r, c in positions:
        allowed[r:r+2, c:c+2] = True
    changed = np.any(emb.image != host, axis=2)
    assert not np.any(changed & ~allowed)


def test_v4_wrong_key_does_not_recover_payload():
    host, wm = _sample(903)
    method = ConvolutionCertifiedR12QIM(_cfg())
    emb = method.embed(host, wm, key=b"v4-right-key")
    wrong = method.extract(emb.image, key=b"v4-wrong-key", side_info=None, watermark_shape=wm.shape)
    assert ber(wm, wrong.watermark) > 0.20
