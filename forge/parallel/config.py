from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ParallelConfig:
    backend: str = "torch"
    dp_mode: str = "fsdp1"
    mixed_precision: str = "bf16"

