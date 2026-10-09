from __future__ import annotations

"""Experimental v5: uniformly conditioned QR + strided Haar-convolution QIM.

This is a research candidate, not a proven improvement in attacked-image BER.
The 2-tap high-pass convolution of column b is (b_top-b_bottom)/sqrt(2).
The other observation is the first QR coefficient r12=q.T@b.  Since
q>=0, ||q||=1, the two filters form a uniformly invertible local map.
The inverse is closed-form and needs no QR decomposition or iterative solver.
"""

from functools import lru_cache
import hashlib
import hmac
import math

import numpy as np

from ..core.config import ProposedConfig
from ..utils.watermark import scrambled_bits_from_watermark, watermark_from_scrambled_bits

_INV_SQRT2 = 1.0 / math.sqrt(2.0)


def _random_below(key: bytes, domain: bytes, index: int, upper: int) -> int:
    """Uniform integer in [0, upper), with rejection to remove modulo bias."""
    limit = (1 << 64) - ((1 << 64) % upper)
    nonce = 0
    while True:
        msg = domain + index.to_bytes(8, 'big') + nonce.to_bytes(4, 'big')
        value = int.from_bytes(hmac.new(key, msg, hashlib.sha256).digest()[:8], 'big')
        if value < limit:
            return value % upper
        nonce += 1


@lru_cache(maxsize=8)
def keyed_sampled_blocks(shape: tuple[int, int], count: int, key: bytes) -> tuple[np.ndarray, np.ndarray]:
    """O(B) HMAC keyed sampling (Floyd), not an O(N)-entry sorted permutation.

    Positions are reproducible under the key, independent of host pixels.
    The ordering uses a separate keyed Fisher-Yates shuffle.
    This is a DIFFERENT carrier mapping from v4 and is not drop-in compatible.
    """
    if not key:
        raise ValueError('nonempty key required')
    h, w = map(int, shape)
    rows, cols = h // 2, w // 2
    total = rows * cols
    if count < 0 or count > total:
        raise ValueError(f'payload {count} exceeds available {total} blocks')
    chosen = set()
    ordered = []
    for j in range(total - count, total):
        cand = _random_below(key, b'convqr:v5:floyd', j, j + 1)
        value = j if cand in chosen else cand
        chosen.add(value)
        ordered.append(value)
    for j in range(count - 1, 0, -1):
        k = _random_below(key, b'convqr:v5:shuffle', j, j + 1)
        ordered[j], ordered[k] = ordered[k], ordered[j]
    indices = np.array(ordered, dtype=np.intp)
    rr = 2 * (indices // cols)
    cc = 2 * (indices % cols)
    rr.flags.writeable = False
    cc.flags.writeable = False
    return rr, cc


def _nearest_centers(x: np.ndarray, bit: np.ndarray, period: float) -> np.ndarray:
    center0 = (0.25 + 0.5 * bit.astype(np.float64)) * period
    return center0 + period * np.rint((x - center0) / period)


def _qim_scores(x: np.ndarray, period: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phase = np.mod(x, period)
    bit = (phase >= 0.5 * period).astype(np.uint8)
    dist = np.minimum(np.minimum(phase, np.abs(phase - 0.5 * period)), period - phase)
    conf = np.clip(4 * dist / period, 0.0, 1.0)
    signed = np.where(bit != 0, np.maximum(conf, np.finfo(np.float64).eps),
                      -np.maximum(conf, np.finfo(np.float64).eps))
    return bit, signed, conf


def _carrier(a0: np.ndarray, a1: np.ndarray, b0: np.ndarray, b1: np.ndarray):
    norm = np.hypot(a0, a1)
    q0 = np.divide(a0, norm, out=np.ones_like(a0), where=norm > 0)
    q1 = np.divide(a1, norm, out=np.zeros_like(a1), where=norm > 0)
    qr = q0 * b0 + q1 * b1
    conv = (b0 - b1) * _INV_SQRT2
    return q0, q1, qr, conv


def _integer_closure(a0, a1, original0, original1, b0, b1, bit, qr_period, conv_period):
    """Bounded repair on exceptional rounded/clipped pairs, no side information.

    Searches nearest integer points to the continuous optimum first; candidates
    satisfying both QIM decisions are selected by minimum squared distortion.
    If no 2-carrier match is available, retain best partial result and report it.
    """
    q0, q1, qr, conv = _carrier(a0, a1, b0.astype(np.float64), b1.astype(np.float64))
    bq, _, _ = _qim_scores(qr, qr_period)
    bh, _, _ = _qim_scores(conv, conv_period)
    bad = np.flatnonzero((bq != bit) | (bh != bit))
    if bad.size == 0:
        return b0, b1, 0

    # First consider neighboring SAME-BIT QIM lattice centers. A saturated
    # pixel can make the nearest continuous target unachievable; another
    # lattice cell can fit with modest distortion without changing any key
    # or storing any side information. Candidate count is constant per block.
    uu = q0[bad] * original0[bad] + q1[bad] * original1[bad]
    vv = (original0[bad] - original1[bad]) * _INV_SQRT2
    center_u = _nearest_centers(uu, bit[bad], qr_period)
    center_v = _nearest_centers(vv, bit[bad], conv_period)
    ky, kh = np.meshgrid(np.arange(-3, 4), np.arange(-3, 4), indexing='ij')
    targets_u = center_u[:, None] + qr_period * ky.ravel()[None, :]
    targets_v = center_v[:, None] + conv_period * kh.ravel()[None, :]
    du = targets_u - uu[:, None]
    dv = targets_v - vv[:, None]
    denom = (q0[bad] + q1[bad])[:, None]
    cand0 = np.clip(np.rint(original0[bad, None] +
                    (du + math.sqrt(2) * q1[bad, None] * dv) / denom), 0, 255)
    cand1 = np.clip(np.rint(original1[bad, None] +
                    (du - math.sqrt(2) * q0[bad, None] * dv) / denom), 0, 255)
    cand_u = q0[bad, None] * cand0 + q1[bad, None] * cand1
    cand_v = (cand0 - cand1) * _INV_SQRT2
    vu, _, _ = _qim_scores(cand_u, qr_period)
    vvbits, _, _ = _qim_scores(cand_v, conv_period)
    feasible = (vu == bit[bad, None]) & (vvbits == bit[bad, None])
    cost = (cand0 - original0[bad, None]) ** 2 + (cand1 - original1[bad, None]) ** 2
    cost[~feasible] = np.inf
    best = np.argmin(cost, axis=1)
    valid = np.isfinite(cost[np.arange(bad.size), best])
    repaired = int(valid.sum())
    if np.any(valid):
        ids = bad[valid]
        chosen = best[valid]
        b0[ids] = cand0[valid, chosen].astype(np.uint8)
        b1[ids] = cand1[valid, chosen].astype(np.uint8)
    bad = bad[~valid]
    if bad.size == 0:
        return b0, b1, repaired
    # 81 candidate pixel pairs. Only remaining mismatch rows are expanded.
    offsets = np.arange(-4, 5, dtype=np.int16)
    du, dv = np.meshgrid(offsets, offsets, indexing='ij')
    c0 = np.clip(b0[bad, None].astype(np.int16) + du.ravel(), 0, 255).astype(np.float64)
    c1 = np.clip(b1[bad, None].astype(np.int16) + dv.ravel(), 0, 255).astype(np.float64)
    u = q0[bad, None] * c0 + q1[bad, None] * c1
    v = (c0 - c1) * _INV_SQRT2
    bu, _, _ = _qim_scores(u, qr_period)
    bv, _, _ = _qim_scores(v, conv_period)
    valid = (bu == bit[bad, None]) & (bv == bit[bad, None])
    costs = (c0 - original0[bad, None]) ** 2 + (c1 - original1[bad, None]) ** 2
    costs[~valid] = np.inf
    choice = np.argmin(costs, axis=1)
    good = np.isfinite(costs[np.arange(bad.size), choice])
    if np.any(good):
        ii = bad[good]
        jj = choice[good]
        b0[ii] = c0[good, jj].astype(np.uint8)
        b1[ii] = c1[good, jj].astype(np.uint8)
    return b0, b1, repaired + int(good.sum())


def embed_convqr_v5_image(host, watermark, key: bytes, cfg: ProposedConfig):
    cfg.validate()
    image = np.asarray(host, dtype=np.uint8)
    wm = np.asarray(watermark, dtype=np.uint8)
    if image.ndim != 3 or image.shape[2] < 3:
        raise ValueError('expected RGB or BGR image')
    if wm.shape != (cfg.watermark_size, cfg.watermark_size):
        raise ValueError('wrong watermark size')
    bits = scrambled_bits_from_watermark(wm, cfg.arnold_iterations)
    channel = image[:, :, cfg.channel]
    rr, cc = keyed_sampled_blocks(channel.shape, int(bits.size), key)
    a0 = channel[rr, cc].astype(np.float64)
    a1 = channel[rr + 1, cc].astype(np.float64)
    b0 = channel[rr, cc + 1].astype(np.float64)
    b1 = channel[rr + 1, cc + 1].astype(np.float64)
    q0, q1, u, v = _carrier(a0, a1, b0, b1)
    pu = float(cfg.convqr_qr_period)
    pv = float(cfg.convqr_conv_period)
    du = _nearest_centers(u, bits, pu) - u
    dv = _nearest_centers(v, bits, pv) - v
    denominator = q0 + q1
    # Exact inverse of [q0 q1; 1/sqrt(2) -1/sqrt(2)] (uniformly conditioned).
    target0 = b0 + (du + math.sqrt(2.0) * q1 * dv) / denominator
    target1 = b1 + (du - math.sqrt(2.0) * q0 * dv) / denominator
    out0 = np.clip(np.rint(target0), 0, 255).astype(np.uint8)
    out1 = np.clip(np.rint(target1), 0, 255).astype(np.uint8)
    out0, out1, repaired = _integer_closure(a0, a1, b0, b1, out0, out1, bits, pu, pv)
    y = image.copy()
    yy = y[:, :, cfg.channel]
    yy[rr, cc + 1] = out0
    yy[rr + 1, cc + 1] = out1
    q0, q1, uf, vf = _carrier(a0, a1, out0.astype(np.float64), out1.astype(np.float64))
    bu, _, _ = _qim_scores(uf, pu)
    bv, _, _ = _qim_scores(vf, pv)
    diff0 = out0.astype(np.float64) - b0
    diff1 = out1.astype(np.float64) - b1
    sse = float(np.sum(diff0 ** 2 + diff1 ** 2))
    meta = {
        'method': 'blind_convqr_v5_experimental', 'fully_blind': True,
        'side_information_bits': 0, 'payload_bits': int(bits.size),
        'qr_bit_match_fraction': float(np.mean(bu == bits)),
        'conv_bit_match_fraction': float(np.mean(bv == bits)),
        'either_bit_match_fraction': float(np.mean((bu == bits) | (bv == bits))),
        'integer_closure_count': repaired,
        'embedding_sse': sse,
        'changed_channel_samples': int(np.count_nonzero(diff0) + np.count_nonzero(diff1)),
        'convolution': 'strided two-tap Haar high-pass on second 2x2 block column',
        'optimality': 'exact unbounded real-valued solution of two linear equality constraints',
        'runtime_path': 'closed_form_sparse_convqr',
    }
    return y, None, meta


def extract_convqr_v5_image(image, key: bytes, cfg: ProposedConfig, watermark_shape=None):
    cfg.validate()
    shape = (cfg.watermark_size, cfg.watermark_size)
    if watermark_shape is not None and tuple(watermark_shape) != shape:
        raise ValueError(f'expected watermark shape {shape}')
    x = np.asarray(image)
    if x.ndim != 3 or x.shape[2] < 3:
        raise ValueError('expected color image')
    ch = x[:, :, cfg.channel]
    rr, cc = keyed_sampled_blocks(ch.shape, shape[0] * shape[1], key)
    a0 = ch[rr, cc].astype(np.float64)
    a1 = ch[rr + 1, cc].astype(np.float64)
    b0 = ch[rr, cc + 1].astype(np.float64)
    b1 = ch[rr + 1, cc + 1].astype(np.float64)
    _, _, u, v = _carrier(a0, a1, b0, b1)
    _, su, cu = _qim_scores(u, cfg.convqr_qr_period)
    _, sv, cv = _qim_scores(v, cfg.convqr_conv_period)
    # Both features are dimensionless confidences with comparable scale.
    combined = su + sv
    bits = (combined >= 0).astype(np.uint8)
    confidence = np.clip(np.abs(combined) * 0.5, 0, 1)
    wm = watermark_from_scrambled_bits(bits, shape[0], cfg.arnold_iterations)
    return wm, float(np.mean(confidence)), {
        'method': 'blind_convqr_v5_experimental', 'fully_blind': True,
        'side_information_used': False, 'original_host_used': False,
        'qr_confidence_mean': float(np.mean(cu)),
        'conv_confidence_mean': float(np.mean(cv)),
        'runtime_path': 'closed_form_sparse_convqr',
    }
