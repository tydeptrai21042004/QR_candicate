from __future__ import annotations

import numpy as np

from qrwatermark.proposed.blind_embedding import _project_safe_intervals
from qrwatermark.proposed.streaming_datapath import project_safe_block_q18, project_safe_selected_pixels
from qrwatermark.utils.watermark import (
    arnold_transform,
    bits_from_watermark,
    inverse_arnold_transform,
    scrambled_bits_from_watermark,
    watermark_from_bits,
    watermark_from_scrambled_bits,
)


def test_bit_domain_arnold_is_exact():
    wm = ((np.arange(64 * 64).reshape(64, 64) % 3) == 0).astype(np.uint8) * 255
    old = bits_from_watermark(arnold_transform(wm, 10))
    new = scrambled_bits_from_watermark(wm, 10)
    assert np.array_equal(old, new)
    assert np.array_equal(
        inverse_arnold_transform(watermark_from_bits(old, (64, 64)), 10),
        watermark_from_scrambled_bits(new, 64, 10),
    )


def test_streaming_safe_projector_matches_block_reference():
    rng = np.random.default_rng(7)
    n = 512
    channel = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    rr = (rng.integers(0, 32, size=n) * 2).astype(np.intp)
    cc = (rng.integers(0, 32, size=n) * 2).astype(np.intp)
    bits = rng.integers(0, 2, size=n, dtype=np.uint8)
    base = np.empty((n, 2, 2), dtype=np.float64)
    base[:, 0, 0] = channel[rr, cc]
    base[:, 0, 1] = channel[rr, cc + 1]
    base[:, 1, 0] = channel[rr + 1, cc]
    base[:, 1, 1] = channel[rr + 1, cc + 1]
    _, rounded, _, _ = _project_safe_intervals(base, bits, 48.0, 0.125)
    y0, y1, *_ = project_safe_selected_pixels(channel, rr, cc, bits, 48.0, 0.125)
    assert np.array_equal(y0, rounded[:, 0, 1].astype(np.uint8))
    assert np.array_equal(y1, rounded[:, 1, 1].astype(np.uint8))


def test_q18_reference_matches_default_projection_on_representative_blocks():
    rng = np.random.default_rng(19)
    blocks = rng.integers(32, 224, size=(512, 2, 2), dtype=np.uint8).astype(np.float64)
    bits = rng.integers(0, 2, size=512, dtype=np.uint8)
    _, rounded, _, _ = _project_safe_intervals(blocks, bits, 48.0, 0.125)
    for i in range(blocks.shape[0]):
        a0 = int(blocks[i, 0, 0]); a1 = int(blocks[i, 1, 0])
        b0 = int(blocks[i, 0, 1]); b1 = int(blocks[i, 1, 1])
        y0, y1 = project_safe_block_q18(a0, a1, b0, b1, int(bits[i]), 18)
        assert y0 == int(rounded[i, 0, 1])
        assert y1 == int(rounded[i, 1, 1])


def test_single_carrier_certificate_kernel_is_exactly_the_general_theorem():
    from qrwatermark.proposed.qr_sensitivity import (
        tightened_two_extreme_convex_group_bound_batch,
        tightened_two_extreme_single_carrier_bound_batch,
    )

    rng = np.random.default_rng(23)
    base = rng.uniform(1.0, 254.0, size=(384, 1, 2, 2))
    ext = np.stack(
        [
            base + rng.normal(0.0, 2.0, size=base.shape),
            base + rng.normal(0.0, 3.0, size=base.shape),
        ],
        axis=0,
    )
    general = tightened_two_extreme_convex_group_bound_batch(
        base, ext, np.ones(1, dtype=np.float64), subdivisions=8
    )
    fast = tightened_two_extreme_single_carrier_bound_batch(
        base[:, 0], ext[:, :, 0], subdivisions=8
    )
    for a, b in zip(general, fast):
        aa = np.asarray(a)
        bb = np.asarray(b)
        if aa.dtype == bool:
            assert np.array_equal(aa, bb)
        else:
            assert np.array_equal(aa, bb)
