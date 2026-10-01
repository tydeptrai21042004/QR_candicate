from __future__ import annotations

from dataclasses import replace

import cv2
import numpy as np
import pytest

from qrwatermark.attacks import apply_attack
from qrwatermark.core.config import BaselineConfig, NLMConfig, ProposedConfig
from qrwatermark.core.interfaces import WatermarkMethod
from qrwatermark.core.types import EmbeddingResult, ExtractionResult
from qrwatermark.evaluation.benchmark import run_benchmark
from qrwatermark.evaluation.evaluator import embed_once, evaluate_embedded
from qrwatermark.evaluation.fairness import tune_baseline_to_psnr
from qrwatermark.proposed import ConvolutionCertifiedR12QIM
from qrwatermark.proposed.certificate import (
    choose_initial_spread_period_for_group,
    theoretical_spread_group_mse,
)
from qrwatermark.proposed.side_info import (
    build_side_info,
    side_info_overhead_bits,
    unpack_certified_mask,
)


class _CountingIdentityMethod(WatermarkMethod):
    name = "counting_identity"

    def __init__(self):
        self.embed_calls = 0
        self._watermark = None

    def embed(self, host: np.ndarray, watermark: np.ndarray, *, key: bytes) -> EmbeddingResult:
        self.embed_calls += 1
        self._watermark = np.asarray(watermark, dtype=np.uint8).copy()
        image = np.asarray(host, dtype=np.uint8).copy()
        # Give the evaluator a finite, deterministic embedding distortion.
        image[0, 0, 0] = np.uint8((int(image[0, 0, 0]) + 1) % 256)
        return EmbeddingResult(image=image, side_info=None, metadata={})

    def extract(
        self,
        image: np.ndarray,
        *,
        key: bytes,
        side_info=None,
        watermark_shape=(64, 64),
    ) -> ExtractionResult:
        return ExtractionResult(watermark=self._watermark.copy(), confidence=1.0)


class _ToyQuantBaseline(WatermarkMethod):
    """Small synthetic baseline used only to regression-test PSNR tuning."""

    name = "toy_quant"

    def __init__(self, config: BaselineConfig | None = None):
        self.config = config or BaselineConfig(quant_step=8.0)
        self._watermark = None

    def embed(self, host: np.ndarray, watermark: np.ndarray, *, key: bytes) -> EmbeddingResult:
        self._watermark = np.asarray(watermark, dtype=np.uint8).copy()
        out = np.asarray(host, dtype=np.float64).copy()
        out[:, :, 0] = np.clip(out[:, :, 0] + float(self.config.quant_step), 0, 255)
        return EmbeddingResult(np.rint(out).astype(np.uint8), None, {})

    def extract(self, image, *, key: bytes, side_info=None, watermark_shape=(64, 64)) -> ExtractionResult:
        return ExtractionResult(self._watermark.copy(), confidence=1.0)


def _smooth_host(size: int = 64) -> np.ndarray:
    yy, xx = np.mgrid[0:size, 0:size]
    base = 70.0 + 0.45 * xx + 0.25 * yy
    return np.stack([base, base + 7, base + 14], axis=2).clip(0, 255).astype(np.uint8)


def _wm(size: int = 4, seed: int = 11) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (rng.integers(0, 2, size=(size, size), dtype=np.uint8) * 255).astype(np.uint8)


def test_initial_period_selection_is_payload_independent():
    rng = np.random.default_rng(101)
    blocks = [rng.uniform(40, 200, size=(2, 2)) for _ in range(3)]
    cfg = ProposedConfig(
        watermark_size=2,
        repetition=3,
        period_candidates=(24.0, 28.0, 32.0, 36.0),
        max_group_mse=1e6,
        target_psnr_db=20.0,
        nlm=NLMConfig(enabled=False),
    )
    d0 = choose_initial_spread_period_for_group(blocks, 0, cfg)
    d1 = choose_initial_spread_period_for_group(blocks, 1, cfg)
    assert d0.period_index == d1.period_index
    assert d0.period == d1.period


def test_initial_period_selection_minimizes_worst_bit_distortion():
    rng = np.random.default_rng(1)
    blocks = [rng.uniform(40, 200, size=(2, 2)) for _ in range(3)]
    cfg = ProposedConfig(
        watermark_size=2,
        repetition=3,
        period_candidates=(24.0, 28.0, 32.0, 36.0),
        max_group_mse=1e6,
        target_psnr_db=20.0,
        nlm=NLMConfig(enabled=False),
    )
    decision = choose_initial_spread_period_for_group(blocks, 0, cfg)
    feasible = []
    for p in cfg.period_candidates:
        mse, ok = theoretical_spread_group_mse(blocks, p)
        if ok and mse <= cfg.max_group_mse:
            feasible.append((mse, p))
    expected = min(feasible, key=lambda x: (x[0], x[1]))[1]
    assert decision.period == expected
    assert decision.period != max(cfg.period_candidates)  # regression for old largest-period bias


def test_impossible_certificate_is_explicitly_marked_uncertified():
    host = _smooth_host(64)
    wm = _wm(4)
    cfg = ProposedConfig(
        watermark_size=4,
        repetition=2,
        period_candidates=(24.0, 28.0, 32.0),
        convolution_gaussian_sigmas=(0.50, 0.75),
        additive_feature_budget=1e4,  # makes Delta/4 survival impossible
        target_psnr_db=35.0,
        certificate_max_passes=3,
        nlm=NLMConfig(enabled=False),
    )
    emb = ConvolutionCertifiedR12QIM(cfg).embed(host, wm, key=b"force-uncertified")
    mask = unpack_certified_mask(emb.side_info, b"force-uncertified")
    assert mask.size == wm.size
    assert not np.any(mask)
    assert emb.metadata["certified_groups"] == 0
    assert emb.metadata["uncertified_groups"] == wm.size
    assert emb.metadata["certificate_reference_signal"] == "final_rounded_watermarked_image"


def test_certificate_metadata_states_post_embedding_reference_and_family():
    host = _smooth_host(64)
    wm = _wm(4, seed=4)
    cfg = ProposedConfig(
        watermark_size=4,
        repetition=2,
        period_candidates=(24.0, 28.0, 32.0, 36.0),
        convolution_gaussian_sigmas=(0.50, 0.75),
        additive_feature_budget=0.0,
        target_psnr_db=35.0,
        nlm=NLMConfig(enabled=False),
    )
    emb = ConvolutionCertifiedR12QIM(cfg).embed(host, wm, key=b"metadata")
    assert emb.metadata["algorithm_revision"] == "mc_ccqr_spread_qim_v4_hw_path_cert"
    assert emb.metadata["certificate_reference_signal"] == "final_rounded_watermarked_image"
    assert "convex mixtures" in emb.metadata["certified_family"]
    assert "rounding_bound" in emb.metadata["certificate_condition"]


def test_side_information_overhead_counts_period_mask_and_authentication_bits():
    codes = np.asarray([0, 1, 2, 3, 0, 1, 2, 3, 0], dtype=np.uint8)
    mask = np.asarray([1, 1, 0, 1, 0, 0, 1, 1, 0], dtype=np.uint8)
    side = build_side_info(codes, {"period_count": 4, "watermark_shape": [3, 3]}, b"overhead", certified_mask=mask)
    overhead = side_info_overhead_bits(side)
    assert overhead["period_code_bits"] == 18  # 9 codes x 2 bits
    assert overhead["certificate_mask_bits"] == 9
    assert overhead["authentication_bits"] == 256
    assert overhead["packed_storage_bits"] == 8 * (3 + 2)
    assert overhead["total_serialized_bits"] >= overhead["packed_storage_bits"] + 256


def test_certified_mask_rejects_wrong_authentication_key():
    codes = np.asarray([0, 1, 0, 1], dtype=np.uint8)
    mask = np.asarray([1, 0, 1, 0], dtype=np.uint8)
    side = build_side_info(codes, {"period_count": 2, "watermark_shape": [2, 2]}, b"right-key", certified_mask=mask)
    with pytest.raises(ValueError, match="authentication"):
        unpack_certified_mask(side, b"wrong-key")


def test_evaluator_separates_embedding_and_attacked_quality_metrics():
    method = _CountingIdentityMethod()
    host = np.full((48, 48, 3), 120, dtype=np.uint8)
    wm = _wm(4)
    emb, elapsed = embed_once(method, host, wm, key=b"eval")
    rec, _ = evaluate_embedded(
        method,
        host,
        wm,
        emb,
        embed_seconds=elapsed,
        key=b"eval",
        host_name="host",
        watermark_name="wm",
        attack_name="gaussian_noise",
        attack_params={"variance": 0.02},
        seed=2,
    )
    assert rec.psnr == rec.embedding_psnr
    assert rec.ssim == rec.embedding_ssim
    assert rec.attacked_psnr < rec.embedding_psnr
    assert rec.attacked_ssim < rec.embedding_ssim


def test_clean_evaluation_has_equal_embedding_and_attacked_metrics():
    method = _CountingIdentityMethod()
    host = np.full((32, 32, 3), 100, dtype=np.uint8)
    wm = _wm(4)
    emb, elapsed = embed_once(method, host, wm, key=b"clean")
    rec, _ = evaluate_embedded(
        method, host, wm, emb, embed_seconds=elapsed, key=b"clean",
        host_name="host", watermark_name="wm", attack_name="clean"
    )
    assert rec.embedding_psnr == rec.attacked_psnr
    assert rec.embedding_ssim == rec.attacked_ssim


def test_benchmark_embeds_once_and_reuses_image_for_all_attacks(tmp_path):
    method = _CountingIdentityMethod()
    host = np.full((32, 32, 3), 110, dtype=np.uint8)
    wm = _wm(8)
    host_path = tmp_path / "host.png"
    wm_path = tmp_path / "wm.png"
    assert cv2.imwrite(str(host_path), host)
    assert cv2.imwrite(str(wm_path), wm)
    attacks = [
        {"name": "clean"},
        {"name": "gaussian_noise", "params": {"variance": 0.001}, "seeds": [1, 2, 3]},
        {"name": "random_pixel_dropout", "params": {"fraction": 0.05}, "seeds": [4, 5]},
    ]
    df = run_benchmark(
        [method], [host_path], [wm_path], attacks,
        key=b"cache", run_dir=tmp_path / "run", save_images=False,
    )
    assert method.embed_calls == 1
    assert len(df) == 6
    assert (tmp_path / "run" / "metrics.csv").exists()


def test_random_pixel_dropout_legacy_alias_is_exactly_reproducible():
    image = np.full((40, 40, 3), 123, dtype=np.uint8)
    new = apply_attack("random_pixel_dropout", image, fraction=0.2, seed=99)
    old = apply_attack("occlusion", image, fraction=0.2, seed=99)
    assert np.array_equal(new, old)


def test_rectangular_occlusion_is_one_contiguous_rectangle():
    image = np.full((50, 60, 3), 200, dtype=np.uint8)
    out = apply_attack("rectangular_occlusion", image, fraction=0.16, seed=7, value=0)
    changed = np.any(out != image, axis=2)
    rows, cols = np.where(changed)
    assert rows.size > 0
    r0, r1, c0, c1 = rows.min(), rows.max(), cols.min(), cols.max()
    assert np.all(changed[r0:r1 + 1, c0:c1 + 1])
    assert int(changed.sum()) == (r1 - r0 + 1) * (c1 - c0 + 1)


def test_registered_rotation_resample_legacy_alias_matches_new_name():
    rng = np.random.default_rng(22)
    image = rng.integers(0, 256, size=(48, 48, 3), dtype=np.uint8)
    new = apply_attack("registered_rotation_resample", image, angle=17.0)
    old = apply_attack("rotation_resample", image, angle=17.0)
    assert np.array_equal(new, old)


def test_certified_convex_gaussian_endpoints_and_mixture_are_finite():
    image = _smooth_host(32)
    y0 = apply_attack("certified_convex_gaussian", image, sigma0=0.50, sigma1=0.75, alpha=0.0, ksize=3)
    y1 = apply_attack("certified_convex_gaussian", image, sigma0=0.50, sigma1=0.75, alpha=1.0, ksize=3)
    ym = apply_attack("certified_convex_gaussian", image, sigma0=0.50, sigma1=0.75, alpha=0.37, ksize=3)
    assert y0.shape == image.shape == y1.shape == ym.shape
    assert y0.dtype == np.float64 and y1.dtype == np.float64 and ym.dtype == np.float64
    assert np.isfinite(ym).all()
    # Linearity in the kernel parameter is a defining property of the certified hull.
    assert np.allclose(ym, (1.0 - 0.37) * y0 + 0.37 * y1, atol=1e-10)


def test_certified_convex_gaussian_rejects_invalid_alpha():
    image = _smooth_host(16)
    with pytest.raises(ValueError, match="alpha"):
        apply_attack("certified_convex_gaussian", image, alpha=1.01)


def test_psnr_tuner_records_target_achieved_strength_and_timing():
    host = np.full((32, 32, 3), 90, dtype=np.uint8)
    wm = _wm(4)
    method = _ToyQuantBaseline(BaselineConfig(quant_step=8.0))
    tuned, emb, diag = tune_baseline_to_psnr(method, host, wm, key=b"tune", target_psnr_db=30.0)
    assert isinstance(tuned, _ToyQuantBaseline)
    assert emb.image.shape == host.shape
    for field in ("target_psnr_db", "achieved_psnr_db", "strength_scale", "quant_step", "embed_seconds", "tuning_seconds", "psnr_error_db"):
        assert field in diag
    assert diag["target_psnr_db"] == 30.0
    assert diag["quant_step"] == pytest.approx(tuned.config.quant_step)
    assert np.isfinite(diag["achieved_psnr_db"])


def test_proposed_config_rejects_nonincreasing_period_table():
    cfg = ProposedConfig(period_candidates=(24.0, 32.0, 28.0))
    with pytest.raises(ValueError, match="strictly increasing"):
        cfg.validate()
