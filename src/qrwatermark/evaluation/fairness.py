from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class ResourceBudget:
    payload_bits: int
    host_pixels: int
    selected_blocks: int | None
    side_information_bits: int | None
    repetition: int | None


def proposed_budget(payload_bits: int, host_pixels: int, repetition: int, period_count: int = 3) -> ResourceBudget:
    bits_per_code=max(1,int(math.ceil(math.log2(period_count))))
    return ResourceBudget(
        payload_bits=payload_bits,
        host_pixels=host_pixels,
        selected_blocks=payload_bits * repetition,
        side_information_bits=payload_bits * bits_per_code,
        repetition=repetition,
    )
