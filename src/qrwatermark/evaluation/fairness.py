from __future__ import annotations

from dataclasses import dataclass, replace
import math
import time

import numpy as np

from .metrics import psnr
from ..proposed.method import GreenQuadWatermark


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
    if isinstance(method, GreenQuadWatermark):
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

    # Refine around the best coarse scale before claiming a PSNR mismatch.
    # This does not use attack outcomes; only embedding distortion is observed.
    best_trial = min(trials, key=lambda x: (x[0], -x[1]))
    if best_trial[0] > 0.05 and len(trials) >= 3:
        unique_scales = sorted({float(t[2]) for t in trials})
        centre = float(best_trial[2]); idx = unique_scales.index(centre)
        lower = unique_scales[max(0,idx-1)]
        upper = unique_scales[min(len(unique_scales)-1,idx+1)]
        if lower < upper:
            for candidate in np.geomspace(lower,upper,9):
                if not any(abs(float(candidate)-float(t[2])) <= 1e-12 for t in trials):
                    trial(float(candidate))
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


def calibrate_global_baseline(method, validation_pairs, *, key: bytes,
                              psnr_tolerance: float = 0.2, scales=None):
    """Freeze one strength setting using ONLY validation image/mark pairs.

    Optimize PSNR matching coverage first, then average absolute PSNR error.
    No attack metric or evaluation/test host is accessed. A baseline with an
    intrinsic distortion floor may remain unmatched; it is not modified.

    validation_pairs: [(host_array, watermark_array, target_psnr), ...]
    """
    pairs = list(validation_pairs)
    if not pairs:
        raise ValueError('Validation host/watermark pairs are required')
    fields = _strength_fields(method)
    if not fields:
        raise ValueError(f'{method.name} has no tunable strength field')
    if scales is None:
        # Limited, preregistered strength sweep; deterministic and attack-blind.
        scales = np.geomspace(0.04, 8.0, 19)
    trials = []
    for scale in scales:
        scale = float(scale)
        try:
            tuned = method.__class__(replace(method.config, **{
                name:float(getattr(method.config,name))*scale for name in fields}))
            errors = np.asarray([float(psnr(h,tuned.embed(h,w,key=key).image))-target
                                 for h,w,target in pairs])
            if not np.all(np.isfinite(errors)):
                raise ValueError('Nonfinite PSNR error')
            match_count = int(np.count_nonzero(np.abs(errors) <= psnr_tolerance))
            trials.append(((-match_count, float(np.mean(np.abs(errors))),
                            float(np.max(np.abs(errors))), scale), tuned, errors))
        except Exception:
            # Some parameter combinations are invalid for a given paper method.
            continue
    if not trials:
        raise ValueError(f'No valid calibration settings for {method.name}')
    # Refine the best valley; a coarse grid alone can miss ±0.2 dB matching.
    best_scale = float(min(trials,key=lambda x:x[0])[0][-1])
    tried = sorted({float(t[0][-1]) for t in trials})
    idx = tried.index(best_scale)
    lo,hi=tried[max(0,idx-1)],tried[min(len(tried)-1,idx+1)]
    if lo < hi:
        for scale in np.geomspace(lo,hi,13):
            scale=float(scale)
            if any(abs(scale-float(t[0][-1]))<1e-12 for t in trials):continue
            try:
                tuned=method.__class__(replace(method.config,**{
                    name:float(getattr(method.config,name))*scale for name in fields}))
                errors=np.asarray([float(psnr(h,tuned.embed(h,w,key=key).image))-target
                                   for h,w,target in pairs])
                if not np.all(np.isfinite(errors)):continue
                score=(-int(np.count_nonzero(np.abs(errors)<=psnr_tolerance)),
                       float(np.mean(np.abs(errors))),float(np.max(np.abs(errors))),scale)
                trials.append((score,tuned,errors))
            except Exception:continue
    score,tuned,errors = min(trials,key=lambda x:x[0])
    scale = float(score[-1])
    return tuned, {
        'method':method.name, 'selected_scale':scale,
        'strength_fields':','.join(fields),
        'validation_pairs':len(pairs),
        'validation_psnr_matched_pairs':int(np.count_nonzero(np.abs(errors)<=psnr_tolerance)),
        'validation_match_fraction':float(np.mean(np.abs(errors)<=psnr_tolerance)),
        'validation_mean_abs_psnr_error_db':float(np.mean(np.abs(errors))),
        'validation_max_abs_psnr_error_db':float(np.max(np.abs(errors))),
        'psnr_tolerance_db':float(psnr_tolerance),
        'settings_tested':len(trials),
        'calibration':'validation-only single global strength; no test-host/attack optimization',
    }
