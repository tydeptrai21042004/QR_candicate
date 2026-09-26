from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..core.config import ProposedConfig
from .r_branch import qim_displacement, qim_displacement_for_block


@dataclass(frozen=True)
class PeriodDecision:
    period_index: int
    period: float
    convolution_shift: float
    required_margin: float
    certificate_margin: float
    worst_case_mse: float
    certified: bool
    distortion_feasible: bool
    range_feasible: bool


def theoretical_group_mse(r12_values: np.ndarray, period: float, blocks: list[np.ndarray] | None = None) -> tuple[float, bool]:
    """Worst-hypothetical-bit exact continuous R12 block MSE.

    With a range-feasible lattice point, changing r12 by delta has exact block
    MSE delta^2/4.  Taking the maximum over hypothetical bits keeps period
    selection independent of the real payload.
    """
    values = np.asarray(r12_values, dtype=np.float64).ravel()
    if blocks is not None and len(blocks) != values.size:
        raise ValueError("blocks and r12_values must have the same length")
    bit_mse = []
    all_range_feasible = True
    for bit in (0, 1):
        deltas = []
        for i, value in enumerate(values):
            if blocks is None:
                delta = qim_displacement(float(value), bit, float(period))
                feasible = True
            else:
                delta, feasible = qim_displacement_for_block(blocks[i], bit, float(period))
            deltas.append(delta)
            all_range_feasible = all_range_feasible and feasible
        d = np.asarray(deltas, dtype=np.float64)
        bit_mse.append(float(np.mean((d * d) / 4.0)))
    return max(bit_mse), bool(all_range_feasible)


def choose_period_for_group(
    r12_values: np.ndarray,
    convolved_r12_values: np.ndarray,
    cfg: ProposedConfig,
    blocks: list[np.ndarray] | None = None,
) -> PeriodDecision:
    """Payload-independent convolution-certificate period selection."""
    base = np.asarray(r12_values, dtype=np.float64).ravel()
    conv = np.asarray(convolved_r12_values, dtype=np.float64)
    if conv.ndim != 2 or conv.shape[1] != base.size:
        raise ValueError("convolved_r12_values must have shape (n_kernels, repetition)")

    conv_shift = 0.0 if conv.shape[0] == 0 else float(np.max(np.abs(conv - base[None, :])))
    required = (
        float(cfg.convolution_safety_factor) * conv_shift
        + float(cfg.additive_feature_budget)
        + float(cfg.rounding_feature_budget)
    )

    decisions: list[PeriodDecision] = []
    for idx, period in enumerate(cfg.period_candidates):
        period = float(period)
        mse, range_ok = theoretical_group_mse(base, period, blocks=blocks)
        residual = period / 4.0 - required
        distortion_ok = mse <= float(cfg.max_group_mse)
        decisions.append(
            PeriodDecision(
                period_index=idx,
                period=period,
                convolution_shift=conv_shift,
                required_margin=required,
                certificate_margin=residual,
                worst_case_mse=mse,
                certified=residual >= 0.0,
                distortion_feasible=distortion_ok,
                range_feasible=range_ok,
            )
        )

    fully_feasible = [d for d in decisions if d.certified and d.distortion_feasible and d.range_feasible]
    if fully_feasible:
        return min(fully_feasible, key=lambda d: (d.worst_case_mse, d.period))

    range_and_dist = [d for d in decisions if d.distortion_feasible and d.range_feasible]
    if range_and_dist:
        return max(range_and_dist, key=lambda d: (d.certificate_margin, -d.worst_case_mse, -d.period))

    distortion_feasible = [d for d in decisions if d.distortion_feasible]
    if distortion_feasible:
        return max(distortion_feasible, key=lambda d: (d.certificate_margin, d.range_feasible, -d.worst_case_mse))

    return min(decisions, key=lambda d: (not d.range_feasible, d.worst_case_mse, d.period))
