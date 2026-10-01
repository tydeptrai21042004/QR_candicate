import numpy as np

from qrwatermark.core.config import NLMConfig, ProposedConfig
from qrwatermark.proposed import ConvolutionCertifiedR12QIM
from qrwatermark.proposed.qr import canonical_qr
from qrwatermark.proposed.qr_sensitivity import (
    convex_hull_group_bound,
    two_extreme_path_group_bound,
    tightened_two_extreme_convex_group_bound,
)
from qrwatermark.proposed.r_branch import apply_r12_delta, qim_phase_decision, r12_value
from qrwatermark.proposed.side_info import unpack_period_indices
from qrwatermark.proposed.spread_qim import group_statistic, spread_decision, unit_spread_weights


def _smooth_host(size=96):
    yy, xx = np.mgrid[0:size, 0:size]
    base = 65.0 + 0.38 * xx + 0.21 * yy + 3.0 * np.sin(xx / 11.0)
    return np.stack([base, base + 6.0, base + 12.0], axis=2).clip(0, 255).astype(np.uint8)


def _watermark(size=8, seed=9):
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=(size, size), dtype=np.uint8) * 255).astype(np.uint8)


def test_closed_form_r12_update_matches_qr_reconstruction_and_rounding():
    rng = np.random.default_rng(2026)
    rounded_mismatches = 0
    max_error = 0.0
    for _ in range(5000):
        block = rng.uniform(8.0, 247.0, size=(2, 2))
        if np.linalg.norm(block[:, 0]) < 1e-6:
            continue
        delta = float(rng.uniform(-10.0, 10.0))
        q, r = canonical_qr(block)
        rr = r.copy()
        rr[0, 1] += delta
        reference = q @ rr
        fast = apply_r12_delta(block, delta)
        max_error = max(max_error, float(np.max(np.abs(reference - fast))))
        rounded_mismatches += int(not np.array_equal(np.rint(reference), np.rint(fast)))
    assert max_error < 1e-10
    assert rounded_mismatches == 0


def test_piecewise_qim_hard_decision_matches_original_sine_rule_away_from_boundaries():
    rng = np.random.default_rng(77)
    period = 36.0
    stats = rng.uniform(-1000.0, 1000.0, size=100000)
    for stat in stats:
        phase = float(stat) % period
        if min(phase, abs(phase - period / 2.0), period - phase) < 1e-8:
            continue
        old_score = np.sin(2.0 * np.pi * (phase - 0.5 * period) / period)
        old_bit = int(old_score >= 0.0)
        bit, score, confidence = qim_phase_decision(float(stat), period)
        assert bit == old_bit
        assert int(score >= 0.0) == bit
        assert 0.0 <= confidence <= 1.0


def test_spread_piecewise_decision_matches_statistic_phase_decision():
    rng = np.random.default_rng(8)
    values = rng.normal(size=5) * 40.0
    weights = unit_spread_weights(5)
    stat = group_statistic(values, weights)
    assert spread_decision(values, 28.0, weights) == qim_phase_decision(stat, 28.0)


def test_two_extreme_piecewise_bound_covers_dense_convex_path():
    rng = np.random.default_rng(88)
    weights = unit_spread_weights(5)
    for _ in range(100):
        base = [rng.uniform(60.0, 190.0, size=(2, 2)) for _ in range(5)]
        e0 = [b + rng.normal(0.0, 2.0, size=(2, 2)) for b in base]
        e1 = [b + rng.normal(0.0, 2.0, size=(2, 2)) for b in base]
        bound, _, _ = two_extreme_path_group_bound(base, [e0, e1], weights, subdivisions=8)
        base_stat = group_statistic(np.asarray([r12_value(b) for b in base]), weights)
        for alpha in np.linspace(0.0, 1.0, 101):
            mixed = [(1.0-alpha)*a + alpha*b for a, b in zip(e0, e1)]
            stat = group_statistic(np.asarray([r12_value(b) for b in mixed]), weights)
            assert abs(stat - base_stat) <= bound + 1e-9


def test_tightened_bound_never_exceeds_generic_bound():
    rng = np.random.default_rng(91)
    base = [rng.uniform(50.0, 200.0, size=(2, 2)) for _ in range(5)]
    e0 = [b + rng.normal(0.0, 3.0, size=(2, 2)) for b in base]
    e1 = [b + rng.normal(0.0, 3.0, size=(2, 2)) for b in base]
    generic, _, _ = convex_hull_group_bound(base, [e0, e1])
    tight, _, _, _, _, _ = tightened_two_extreme_convex_group_bound(base, [e0, e1], subdivisions=8)
    assert tight <= generic + 1e-12


def test_final_path_tightening_changes_only_certificate_not_embedded_pixels_or_periods():
    host = _smooth_host(96)
    wm = _watermark(8)
    common = dict(
        watermark_size=8,
        repetition=3,
        period_candidates=(24.0, 28.0, 32.0, 36.0),
        convolution_gaussian_sigmas=(0.50, 0.75),
        additive_feature_budget=0.0,
        target_psnr_db=42.0,
        certificate_max_passes=4,
        certificate_path_subdivisions=8,
        nlm=NLMConfig(enabled=False),
    )
    key = b"v4-no-pixel-change"
    generic = ConvolutionCertifiedR12QIM(ProposedConfig(**common, certificate_final_tighten=False)).embed(host, wm, key=key)
    tight = ConvolutionCertifiedR12QIM(ProposedConfig(**common, certificate_final_tighten=True)).embed(host, wm, key=key)
    assert np.array_equal(generic.image, tight.image)
    generic_periods, _ = unpack_period_indices(generic.side_info, key)
    tight_periods, _ = unpack_period_indices(tight.side_info, key)
    assert np.array_equal(generic_periods, tight_periods)
    assert tight.metadata["actual_rgb_sse"] == generic.metadata["actual_rgb_sse"]
    assert tight.metadata["actual_psnr_db"] == generic.metadata["actual_psnr_db"]
    assert tight.metadata["certified_groups"] >= generic.metadata["certified_groups"]
