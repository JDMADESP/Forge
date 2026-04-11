from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn


@dataclass(frozen=True)
class ConditionSchema:
    required_fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = ()
    forward_arg_map: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ArchitectureParallelSpec:
    wrap_block_classes: tuple[str, ...] = ()
    no_shard_modules: tuple[str, ...] = ()
    shardable_inputs: dict[str, int] = field(default_factory=dict)
    replicate_inputs: tuple[str, ...] = ()


class ModelArchitecture(ABC):
    """Static model-family definition."""

    condition_schema: ConditionSchema
    parallel_spec: ArchitectureParallelSpec

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build an uninitialized model instance."""

    def remap_state_dict(self, state_dict: dict[str, Any]) -> dict[str, Any]:
        return state_dict

