from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

import numpy as np

from .metrics import psnr
from ..proposed.method import ConvolutionCertifiedR12QIM


@dataclass(frozen=True)
class ResourceBudget:
    payload_bits: int
    host_pixels: int
    selected_blocks: int | None
    side_information_bits: int | None
    repetition: int | None


def proposed_budget(
    payload_bits: int,
    host_pixels: int,
    repetition: int,
    period_count: int = 4,
    include_certificate_mask: bool = True,
) -> ResourceBudget:
    bits_per_code = max(1, int(math.ceil(math.log2(period_count))))
    side = payload_bits * bits_per_code + (payload_bits if include_certificate_mask else 0) + 256
    return ResourceBudget(payload_bits, host_pixels, payload_bits * repetition, side, repetition)


def _strength_fields(method) -> tuple[str, ...]:
    fields = getattr(method, "strength_fields", None)
    if fields:
        return tuple(str(x) for x in fields)
    field = getattr(method, "strength_field", None)
    if field:
        return (str(field),)
    # Backward-compatible fallback for local/custom baselines.
    if hasattr(method, "config") and hasattr(method.config, "quant_step"):
        return ("quant_step",)
    return ()


def tune_baseline_to_psnr(
    method,
    host,
    watermark,
    *,
    key: bytes,
    target_psnr_db: float,
    scales=None,
):
    """Tune each baseline's *actual published strength parameter(s)* to PSNR.

    Single-parameter papers expose ``strength_field``.  Multi-parameter papers
    (Su-Zhang-Wang 2020) expose ``strength_fields`` and all listed parameters are
    multiplied by the same dimensionless scale, preserving their paper ratio.

    The routine returns the closest achievable PSNR and explicitly reports the
    residual rather than pretending a method with a distortion floor matched
    the proposal exactly.
    """
    if isinstance(method, ConvolutionCertifiedR12QIM):
        emb = method.embed(host, watermark, key=key)
        p = float(psnr(host, emb.image))
        return method, emb, {
            "target_psnr_db": float(target_psnr_db),
            "achieved_psnr_db": p,
            "strength_scale": 1.0,
            "psnr_error_db": float(p - target_psnr_db),
        }

    fields = _strength_fields(method)
    if not fields or not hasattr(method, "config"):
        emb = method.embed(host, watermark, key=key)
        p = float(psnr(host, emb.image))
        return method, emb, {
            "target_psnr_db": float(target_psnr_db),
            "achieved_psnr_db": p,
            "strength_scale": 1.0,
            "psnr_error_db": float(p - target_psnr_db),
        }

    base_values: dict[str, float] = {}
    for field in fields:
        if not hasattr(method.config, field):
            raise AttributeError(f"{method.name} declares missing strength field {field!r}")
        value = float(getattr(method.config, field))
        if value <= 0.0:
            raise ValueError(f"{method.name}.{field} must be positive for PSNR matching")
        base_values[field] = value

    auto_scales = scales is None
    # Include substantially weaker settings: some paper defaults target ~35-41
    # dB whereas this proposal often operates near/above 50 dB.
    scales = np.asarray(scales if scales is not None else np.geomspace(0.05, 8.0, 11), dtype=np.float64)
    trials = []
    total_t0 = time.perf_counter()

    def trial(scale: float) -> None:
        scale = float(scale)
        kwargs = {field: base_values[field] * scale for field in fields}
        tuned = method.__class__(replace(method.config, **kwargs))
        t0 = time.perf_counter()
        emb = tuned.embed(host, watermark, key=key)
        embed_s = time.perf_counter() - t0
        p = float(psnr(host, emb.image))
        trials.append((abs(p - float(target_psnr_db)), p, scale, tuned, emb, float(embed_s)))

    for scale in scales:
        trial(float(scale))

    if auto_scales:
        best = min(trials, key=lambda x: (x[0], -x[1]))
        center = float(best[2])
        lo = max(0.005, 0.65 * center)
        hi = 1.35 * center
        for scale in np.linspace(lo, hi, 7):
            if not any(abs(float(scale) - float(t[2])) <= 1e-12 for t in trials):
                trial(float(scale))

    _, p, scale, tuned, emb, embed_s = min(trials, key=lambda x: (x[0], -x[1]))
    diag = {
        "target_psnr_db": float(target_psnr_db),
        "achieved_psnr_db": float(p),
        "strength_scale": float(scale),
        "strength_fields": ",".join(fields),
        "embed_seconds": float(embed_s),
        "tuning_seconds": float(time.perf_counter() - total_t0),
        "psnr_error_db": float(p - target_psnr_db),
    }
    for field in fields:
        diag[field] = float(getattr(tuned.config, field))
    return tuned, emb, diag
