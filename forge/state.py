from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class BatchState:
    """Mutable step state shared across the training pipeline."""

    raw_batch: dict[str, Any]
    model_inputs: dict[str, Any] = field(default_factory=dict)
    targets: dict[str, Any] = field(default_factory=dict)
    loss_dict: dict[str, Any] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)
    artifacts: dict[str, Any] = field(default_factory=dict)
