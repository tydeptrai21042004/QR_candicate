"""Regression tests for the post-repair deterministic carrier mapping.

These assert internal consistency, not equivalence with the absent original v6.
"""
import numpy as np
import pytest

from qrwatermark.proposed.block_carrier import keyed_blocks_v6, _gather_blocks


def test_carriers_are_keyed_deterministic_and_disjoint():
    a = keyed_blocks_v6((512, 512), 4096, b"constant-key")
    b = keyed_blocks_v6((512, 512), 4096, b"constant-key")
    c = keyed_blocks_v6((512, 512), 4096, b"different-key")
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert any(not np.array_equal(x, y) for x, y in zip(a, c))
    rr, cc = a
    assert np.all(rr % 4 == 0) and np.all(cc % 4 == 0)
    assert len(set(zip(rr.tolist(), cc.tolist()))) == 4096
    assert np.all((rr >= 0) & (rr + 3 < 512))
    assert np.all((cc >= 0) & (cc + 3 < 512))


def test_gather_matches_individual_slices():
    image = np.arange(25 * 26).reshape(25, 26)
    rr, cc = keyed_blocks_v6(image.shape, 6, b"key")
    blocks = _gather_blocks(image, rr, cc)
    assert blocks.shape == (6, 4, 4)
    for k in range(6):
        assert np.array_equal(blocks[k], image[rr[k]:rr[k]+4, cc[k]:cc[k]+4])


def test_capacity_and_key_checks():
    with pytest.raises(ValueError):
        keyed_blocks_v6((8, 8), 5, b"key")
    with pytest.raises(ValueError):
        keyed_blocks_v6((8, 8), -1, b"key")
    with pytest.raises(TypeError):
        keyed_blocks_v6((8, 8), 1, "not-bytes")
    with pytest.raises(ValueError):
        _gather_blocks(np.ones((4, 4, 3)), np.array([0]), np.array([0]))
