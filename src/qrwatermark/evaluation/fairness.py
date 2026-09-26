from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ResourceBudget:
    payload_bits: int
    host_pixels: int
    selected_blocks: int | None
    side_information_bits: int | None
    repetition: int | None


def proposed_budget(payload_bits: int, host_pixels: int, repetition: int) -> ResourceBudget:
    return ResourceBudget(
        payload_bits=payload_bits,
        host_pixels=host_pixels,
        selected_blocks=payload_bits * repetition,
        side_information_bits=payload_bits,  # one mode bit per payload bit in v2
        repetition=repetition,
    )
