"""Experimental eight-pixel integer low-pass / constrained-QR carrier.

The normalized eight-pixel sum S/sqrt(8) is the cross coefficient of a
fixed-direction QR factor.  Because the direction is constant, the online
implementation needs only integer addition, integer-QIM, and an exact integer
inverse.  It is a low-pass carrier, NOT a novel general-purpose QR algorithm.

No attack-specific branches, ECC, repeated payload bits, alignment pilots,
original host at extraction, or transmitted side information are used.
"""
from __future__ import annotations

import numpy as np

from .block_carrier import _gather_blocks, keyed_blocks_v6
from ..utils.watermark import scrambled_bits_from_watermark, watermark_from_scrambled_bits


def _lattice_centres(sums: np.ndarray, bits: np.ndarray, period: int, max_sum: int = 2040) -> np.ndarray:
    """Nearest feasible integer QIM centre for 8 uint8 pixels."""
    centre = period // 4 + (period // 2) * bits.astype(np.int32)
    index = np.rint((sums.astype(np.float64) - centre) / period).astype(np.int32)
    max_index = (max_sum - centre) // period
    return centre + period * np.maximum(0, np.minimum(index, max_index))


def _distribute_exact(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Exact-sum integer update; rare saturated rows use bounded repair.

    For unsaturated pixels x'_i=x_i+q+[i<r], where target-S=8*q+r.
    If a pixel would leave [0,255], redistribute the leftover only among
    coordinates with remaining capacity.  This is deterministic and depends
    only on pixel capacity, not on attack type.
    """
    v = values.astype(np.int16, copy=False)
    diff = targets.astype(np.int32) - v.astype(np.int32).sum(axis=1)
    count = v.shape[1]
    q, r = np.divmod(diff, count)
    raw = v.astype(np.int32) + q[:, None] + (np.arange(count)[None, :] < r[:, None])
    bad = np.flatnonzero(np.any((raw < 0) | (raw > 255), axis=1))
    output = np.clip(raw, 0, 255).astype(np.int16)

    # Saturation repair normally touches no rows.  Only a few coordinates
    # can be saturated, and each loop completes an equal-allocation pass.
    if bad.size:
        repair = output[bad].copy()
        residual = targets[bad].astype(np.int32) - repair.astype(np.int32).sum(axis=1)
        for _ in range(9):
            if not np.any(residual):
                break
            positive = residual > 0
            capacity = np.where(positive[:, None], 255 - repair, repair)
            active = capacity > 0
            n_active = active.sum(axis=1)
            if np.any((residual != 0) & (n_active == 0)):
                raise ArithmeticError('QIM target outside feasible pixel range')
            # Equal share followed by deterministic remainder distribution.
            share = np.abs(residual) // np.maximum(n_active, 1)
            change = np.minimum(capacity, share[:, None]) * active
            repair += change.astype(np.int16) * np.where(positive, 1, -1)[:, None]
            residual = targets[bad].astype(np.int32) - repair.astype(np.int32).sum(axis=1)
            positive = residual > 0
            capacity = np.where(positive[:, None], 255 - repair, repair)
            active = capacity > 0
            rank = np.cumsum(active, axis=1)
            one = active & (rank <= np.abs(residual)[:, None])
            repair += one.astype(np.int16) * np.where(positive, 1, -1)[:, None]
            residual = targets[bad].astype(np.int32) - repair.astype(np.int32).sum(axis=1)
        if np.any(residual):
            raise ArithmeticError('exact integer sum correction did not converge')
        output[bad] = repair
    if np.any(output.astype(np.int32).sum(axis=1) != targets):
        raise ArithmeticError('integer QIM closure failed')
    return output.astype(np.uint8)


def _check_image_and_shape(image, wm_shape):
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError('expected a three-channel uint8 host')
    if image.dtype != np.uint8:
        raise ValueError('expected uint8 image')
    if wm_shape[0] != wm_shape[1]:
        raise ValueError('square watermark required for Arnold scrambling')


def embed_integer_conv8_image(host, watermark, key, cfg):
    cfg.validate()
    image = np.asarray(host)
    wm = np.asarray(watermark, dtype=np.uint8)
    _check_image_and_shape(image, wm.shape)
    expected = (cfg.watermark_size, cfg.watermark_size)
    if wm.shape != expected:
        raise ValueError(f'expected watermark shape {expected}')
    bits = scrambled_bits_from_watermark(wm, cfg.arnold_iterations).ravel()
    rr, cc = keyed_blocks_v6(image.shape[:2], int(bits.size), key)
    patches = _gather_blocks(image[:, :, cfg.channel], rr, cc)
    # Right half of 4x4 = two disjoint 2x2 normalized box-filter cells.
    eight = patches[:, :, 2:].reshape(-1, 8)
    sums = eight.astype(np.int16).sum(axis=1).astype(np.int32)
    period = int(cfg.integer_conv_period)
    targets = _lattice_centres(sums, bits, period)
    updated = _distribute_exact(eight, targets)
    modified = patches.copy()
    modified[:, :, 2:] = updated.reshape(-1, 4, 2)
    output = image.copy()
    channel = output[:, :, cfg.channel]
    height, width = channel.shape
    channel[:height//4*4, :width//4*4].reshape(height//4, 4, width//4, 4)[rr//4, :, cc//4, :] = modified
    return output, None, {
        'method': 'blind_integer_convolution_v7_experimental',
        'fully_blind': True,
        'side_information_bits': 0,
        'payload_bits': int(bits.size),
        'clean_integer_failures': 0,
        'integer_conv_period': period,
        'convolution': '8 pixels; two disjoint 2x2 low-pass supports',
        'runtime_path': 'integer_sum_fixed_direction_qr',
    }


def extract_integer_conv8_image(image, key, cfg, watermark_shape=None):
    cfg.validate()
    x = np.asarray(image)
    expected = (cfg.watermark_size, cfg.watermark_size)
    if watermark_shape is not None and tuple(watermark_shape) != expected:
        raise ValueError(f'expected watermark shape {expected}')
    _check_image_and_shape(x, expected)
    rr, cc = keyed_blocks_v6(x.shape[:2], expected[0] * expected[1], key)
    patches = _gather_blocks(x[:, :, cfg.channel], rr, cc)
    total = patches[:, :, 2:].reshape(-1, 8).astype(np.int16).sum(axis=1)
    period = int(cfg.integer_conv_period)
    remainder = total % period
    bits = (remainder >= period//2).astype(np.uint8)
    confidence = float(np.mean(np.minimum(remainder % (period//2), period//2 - (remainder % (period//2))) / (period/4)))
    recovered = watermark_from_scrambled_bits(bits, expected[0], cfg.arnold_iterations)
    return recovered, confidence, {
        'method': 'blind_integer_convolution_v7_experimental',
        'fully_blind': True,
        'original_host_used': False,
        'side_information_used': False,
        'runtime_path': 'integer_sum_fixed_direction_qr',
    }
