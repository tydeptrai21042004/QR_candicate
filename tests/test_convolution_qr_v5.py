"""Experimental convolutional QR carrier: reproducibility and exact math."""

import numpy as np
import pytest

from qrwatermark.core.config import ProposedConfig
from qrwatermark.proposed import ConvolutionCertifiedR12QIM
from qrwatermark.proposed.convolution_qr_v5 import (
    keyed_sampled_blocks, _carrier, _nearest_centers,
)
from qrwatermark.evaluation.metrics import ber


def _method(n=8, p=6.0):
    return ConvolutionCertifiedR12QIM(ProposedConfig(
        design='convqr_v5', watermark_size=n, convqr_qr_period=p,
        convqr_conv_period=p, compute_certificate=False,
        store_certificate_mask=False,
    ))


@pytest.mark.parametrize('seed', range(100))
def test_100_random_clean_roundtrips(seed):
    rng = np.random.default_rng(seed)
    host = rng.integers(12, 244, size=(64, 64, 3), dtype=np.uint8)
    wm = rng.integers(0, 2, size=(8, 8), dtype=np.uint8) * 255
    method = _method()
    key = f'v5-seed-{seed}'.encode()
    result = method.embed(host, wm, key=key)
    decoded = method.extract(result.image, key=key, watermark_shape=wm.shape)
    assert result.side_info is None
    assert result.metadata['side_information_bits'] == 0
    assert result.metadata['fully_blind']
    assert ber(wm, decoded.watermark) == 0


def test_uniform_condition_number_sharp():
    for theta in np.linspace(0.0, np.pi / 2, 10001):
        q0, q1 = np.cos(theta), np.sin(theta)
        mat = np.array([[q0, q1], [1 / np.sqrt(2), -1 / np.sqrt(2)]])
        assert np.linalg.cond(mat) <= 1.0 + np.sqrt(2) + 1e-12
    assert abs(np.linalg.cond(np.array([[1, 0], [2**-.5, -2**-.5]])) - (1 + 2**.5)) < 1e-12


def test_closed_form_inverse_exact_for_random_real_values():
    rng = np.random.default_rng(72)
    a0, a1 = rng.uniform(1, 255, size=(2, 500))
    b0, b1 = rng.uniform(0, 255, size=(2, 500))
    bits = rng.integers(0, 2, size=500, dtype=np.uint8)
    q0, q1, u, v = _carrier(a0, a1, b0, b1)
    du = _nearest_centers(u, bits, 6) - u
    dv = _nearest_centers(v, bits, 6) - v
    db0 = (du + np.sqrt(2) * q1 * dv) / (q0 + q1)
    db1 = (du - np.sqrt(2) * q0 * dv) / (q0 + q1)
    _, _, uu, vv = _carrier(a0, a1, b0 + db0, b1 + db1)
    np.testing.assert_allclose(uu, u + du, atol=2e-13)
    np.testing.assert_allclose(vv, v + dv, atol=1e-13)


def test_v5_changes_only_selected_second_column():
    rng = np.random.default_rng(15)
    host = rng.integers(8, 247, size=(64, 64, 3), dtype=np.uint8)
    wm = rng.integers(0, 2, size=(8, 8), dtype=np.uint8) * 255
    key = b'convqr-positions'
    emb = _method().embed(host, wm, key=key)
    rr, cc = keyed_sampled_blocks((64, 64), wm.size, key)
    mask = np.zeros(host.shape, dtype=bool)
    mask[rr, cc + 1, 0] = True
    mask[rr + 1, cc + 1, 0] = True
    assert np.array_equal(host[~mask], emb.image[~mask])
    assert len(set(zip(rr.tolist(), cc.tolist()))) == wm.size


def test_v5_wrong_key_and_repeatability():
    rng = np.random.default_rng(1911)
    host = rng.integers(10, 245, size=(64, 64, 3), dtype=np.uint8)
    wm = rng.integers(0, 2, size=(8, 8), dtype=np.uint8) * 255
    m = _method()
    emb = m.embed(host, wm, key=b'correct')
    assert np.array_equal(emb.image, m.embed(host, wm, key=b'correct').image)
    wrong = m.extract(emb.image, key=b'wrong', watermark_shape=wm.shape)
    assert ber(wm, wrong.watermark) > 0.20


def test_capacity_rejected():
    with pytest.raises(ValueError, match='exceeds'):
        keyed_sampled_blocks((8, 8), 32, b'overflow')


@pytest.mark.parametrize('constant', [0, 1, 254, 255])
def test_saturated_uniform_images_roundtrip(constant):
    rng = np.random.default_rng(123)
    host = np.full((64, 64, 3), constant, dtype=np.uint8)
    wm = rng.integers(0, 2, (8, 8), dtype=np.uint8) * 255
    method = _method(n=8, p=24.0)
    emb = method.embed(host, wm, key=b'constant-extreme')
    decoded = method.extract(emb.image, key=b'constant-extreme', watermark_shape=wm.shape)
    assert ber(wm, decoded.watermark) == 0.0


def test_factory_integration():
    from pathlib import Path
    from qrwatermark.core.factory import build_method
    root = Path(__file__).resolve().parents[1]
    method = build_method('proposed', root / 'configs/methods/proposed_convqr_v5_experimental.yaml')
    assert method.name == 'blind_convqr_v5_experimental'
    assert method.config.convqr_qr_period == 24.0
