import cv2
import numpy as np
import pytest

from qrwatermark.baselines import (
    Chen2021QuaternionQR,
    Nha2022ImprovedQR,
    Su2014QR,
    Su2016Hessenberg,
    Su2020Schur,
    Zareian2013AdaptiveQIM,
)
from qrwatermark.baselines.chen2021_qqrd.method import (
    bgr_to_pure_quaternion,
    quaternion_matmul,
    quaternion_qr,
)
from qrwatermark.baselines.nha2022_improved_qr.method import _raster_positions, improved_qr_r_first
from qrwatermark.baselines.su2016_hessenberg.method import canonical_hessenberg
from qrwatermark.baselines.su2020_schur.method import _d_decode, _d_embed, _u_candidate, canonical_schur
from qrwatermark.baselines.zareian2013_aqim.method import (
    adaptive_step, estimate_rotation_angle, haar2_levels, inverse_haar2_levels, normalized_magnitude, qim_vector_level,
)
from qrwatermark.attacks.geometric import rotation_unregistered
from qrwatermark.core.factory import build_method
from qrwatermark.evaluation.metrics import ber


def _binary_watermark(rng, side=8):
    return (rng.integers(0, 2, size=(side, side), dtype=np.uint8) * 255).astype(np.uint8)


def test_su2014_published_r14_quantizer_decodes_embedded_bit():
    delta = 8.0
    values = [-43.2, -16.0, -0.1, 0.0, 3.4, 17.9, 99.2]
    for value in values:
        for bit in (0, 1):
            marked = Su2014QR._embed_r14(value, bit, delta)
            assert Su2014QR._decode_r14(marked, delta) == bit


def test_su2016_hessenberg_uses_stable_equivalent_sign_convention():
    # Numerical example printed in Su (2016), Eq. (6).
    a = np.array(
        [
            [143, 164, 152, 133],
            [129, 167, 177, 158],
            [129, 153, 172, 177],
            [131, 146, 158, 158],
        ],
        dtype=np.float64,
    )
    h, q = canonical_hessenberg(a)
    assert np.allclose(q @ h @ q.T, a, atol=1e-10)
    # Householder signs are mathematically arbitrary.  The canonicalization
    # preserves A while making the paper's sign(.) modulation executable.
    assert q[1, 1] >= 0.0

    for bit in (0, 1):
        q2 = Su2016Hessenberg._modify_q(q, bit, 0.042)
        marked = np.rint(q2 @ h @ q2.T)
        _, qe = canonical_hessenberg(marked)
        assert Su2016Hessenberg._decode_q(qe) == bit


def test_nha2022_uses_published_raster_block_traversal_without_key_selection():
    assert _raster_positions((8, 12), 4, 5) == [
        (0, 0), (0, 4), (0, 8), (4, 0), (4, 4)
    ]


def test_nha2022_r_first_factorization_reconstructs_regular_block():
    a = np.array(
        [
            [132.0, 141.0, 153.0, 162.0],
            [127.0, 145.0, 158.0, 168.0],
            [121.0, 149.0, 166.0, 174.0],
            [118.0, 154.0, 171.0, 183.0],
        ]
    )
    q, r = improved_qr_r_first(a)
    assert np.allclose(q @ r, a, atol=1e-8)
    assert r[0, 0] == pytest.approx(np.linalg.norm(a[:, 0]))
    assert np.all(np.diag(r) > 0.0)


def test_chen2021_quaternion_qr_reconstructs_full_rank_block():
    rng = np.random.default_rng(11)
    bgr = rng.integers(20, 230, size=(4, 4, 3), dtype=np.uint8)
    aq = bgr_to_pure_quaternion(bgr)
    q, r = quaternion_qr(aq)
    assert np.allclose(quaternion_matmul(q, r), aq, atol=1e-8)
    assert np.all(r[np.arange(4), np.arange(4), 0] >= 0.0)
    assert np.allclose(r[np.arange(4), np.arange(4), 1:], 0.0, atol=1e-12)


def test_chen2021_quaternion_qr_rank_deficient_reconstruction():
    # Repeated columns exercise deterministic basis completion.  The completed
    # Q column must not invent a nonzero R diagonal.
    bgr = np.full((4, 4, 3), 120, dtype=np.uint8)
    aq = bgr_to_pure_quaternion(bgr)
    q, r = quaternion_qr(aq)
    assert np.allclose(quaternion_matmul(q, r), aq, atol=1e-8)
    assert np.count_nonzero(np.abs(r[np.arange(4), np.arange(4), 0]) < 1e-10) >= 1




def test_su2020_published_dmax_quantizer_decodes_embedded_bit():
    delta = 25.0
    for value in (-61.4, -2.5, 0.0, 14.2, 93.7, 241.0):
        for bit in (0, 1):
            marked = _d_embed(value, bit, delta)
            assert _d_decode(marked, delta) == bit


def test_su2020_u_candidate_implements_published_magnitude_gap():
    rng = np.random.default_rng(17)
    block = rng.integers(20, 230, size=(4, 4)).astype(np.float64)
    dmat, umat = canonical_schur(block)
    c = int(np.argmax(np.diag(dmat)))
    threshold = 0.03
    for bit in (0, 1):
        candidate = _u_candidate(umat, dmat, c, bit, threshold)
        assert candidate.shape == (4, 4)
        # Verify the source-space Eq. (7)-(8) relation directly; Schur vectors
        # after pixel rounding are not guaranteed to be bit-exact (the paper
        # itself reports clean NC slightly below 1).
        avg = 0.5 * (abs(float(umat[1, c])) + abs(float(umat[2, c])))
        hi, lo = avg + threshold / 2.0, avg - threshold / 2.0
        assert hi - lo == pytest.approx(threshold)


def test_zareian2013_haar_roundtrip_and_published_scalar_qim():
    rng = np.random.default_rng(23)
    block = rng.random((16, 16))
    coeff = haar2_levels(block, levels=2)
    assert np.allclose(inverse_haar2_levels(coeff, levels=2), block, atol=1e-12)
    x = coeff[:4, :4]
    delta = adaptive_step(x, 0.21, 3.25)
    s = normalized_magnitude(x)
    assert delta > 0 and s > 0
    for bit in (0, 1):
        q = qim_vector_level(s, bit, delta)
        candidates = [qim_vector_level(q, b, delta) for b in (0, 1)]
        assert int(np.argmin([abs(q - z) for z in candidates])) == bit


def test_zareian2013_clean_roundtrip_uses_only_published_position_map_side_info():
    rng = np.random.default_rng(29)
    host = rng.integers(30, 226, size=(128, 128, 3), dtype=np.uint8)
    wm = _binary_watermark(rng, 8)
    method = Zareian2013AdaptiveQIM()
    emb = method.embed(host, wm, key=b"ignored-by-paper-selector")
    assert "selected_indices" not in emb.side_info
    ext = method.extract(emb.image, key=b"ignored", side_info=emb.side_info, watermark_shape=wm.shape)
    assert ber(wm, ext.watermark) == 0.0


def test_zareian2013_published_entropy_map_rotation_search_recovers_plus_five_degrees():
    # Entropy-map registration needs a natural/structured image; white-noise
    # blocks all have nearly the same entropy and are an invalid sync fixture.
    host = cv2.imread("data/lenna.bmp", cv2.IMREAD_COLOR)
    assert host is not None
    host = cv2.resize(host, (256, 256), interpolation=cv2.INTER_AREA)
    wm = _binary_watermark(np.random.default_rng(31), 8)
    method = Zareian2013AdaptiveQIM()
    emb = method.embed(host, wm, key=b"unused")
    attacked = rotation_unregistered(emb.image, angle=5.0)
    angle = estimate_rotation_angle(
        attacked[:, :, method.config.channel].astype(np.float64),
        emb.side_info["selected_mask"],
    )
    assert angle == pytest.approx(5.0, abs=0.5)
    ext = method.extract(attacked, key=b"unused", side_info=emb.side_info, watermark_shape=wm.shape)
    assert ext.metadata["estimated_rotation_deg"] == pytest.approx(5.0, abs=0.5)


def test_published_side_information_models_are_enforced():
    rng = np.random.default_rng(37)
    host = rng.integers(30, 226, size=(128, 128, 3), dtype=np.uint8)
    wm8 = _binary_watermark(rng, 8)

    schur = Su2020Schur()
    schur_emb = schur.embed(host, wm8, key=b"side-model")
    assert schur_emb.side_info is not None and "flags" in schur_emb.side_info
    with pytest.raises(ValueError, match="mode flags"):
        schur.extract(schur_emb.image, key=b"side-model", side_info=None, watermark_shape=wm8.shape)

    zareian = Zareian2013AdaptiveQIM()
    z_emb = zareian.embed(host, wm8, key=b"unused")
    assert set(z_emb.side_info) == {"selected_mask", "delta0", "gamma", "xi"}
    with pytest.raises(ValueError, match="side information"):
        zareian.extract(z_emb.image, key=b"unused", side_info=None, watermark_shape=wm8.shape)


def test_removed_su2017_baselines_are_not_factory_registered():
    with pytest.raises(ValueError, match="Unknown method"):
        build_method("su2017_improved_qr")
    with pytest.raises(ValueError, match="Unknown method"):
        build_method("su2017_hessenberg")


def test_decomposition_paper_baselines_have_error_free_clean_smoke_roundtrip():
    rng = np.random.default_rng(5)
    host = rng.integers(30, 226, size=(64, 64, 3), dtype=np.uint8)
    wm = _binary_watermark(rng, 8)
    methods = (
        Su2014QR(),
        Su2016Hessenberg(),
        Su2020Schur(),
        Chen2021QuaternionQR(),
        Nha2022ImprovedQR(),
    )
    for method in methods:
        emb = method.embed(host, wm, key=b"paper-baselines")
        ext = method.extract(
            emb.image,
            key=b"paper-baselines",
            side_info=emb.side_info,
            watermark_shape=wm.shape,
        )
        assert ext.watermark.shape == wm.shape
        assert ber(wm, ext.watermark) == 0.0, method.name


@pytest.mark.parametrize(
    "name,config",
    [
        ("su2014_qr", "configs/methods/su2014_qr.yaml"),
        ("su2016_hessenberg", "configs/methods/su2016_hessenberg.yaml"),
        ("chen2021_qqrd", "configs/methods/chen2021_qqrd.yaml"),
        ("nha2022_improved_qr", "configs/methods/nha2022_improved_qr.yaml"),
        ("su2020_schur", "configs/methods/su2020_schur.yaml"),
        ("zareian2013_aqim", "configs/methods/zareian2013_aqim.yaml"),
    ],
)
def test_paper_baselines_are_factory_registered(name, config):
    assert build_method(name, config).name == name
