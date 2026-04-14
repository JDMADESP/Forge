from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch.nn as nn

from forge.architectures.base import ModelArchitecture
from forge.batch import DenoiseBatch
from forge.parallel.config import ParallelConfig


class ModelRuntime(ABC):
    """Dynamic model-family runtime."""

    architecture: ModelArchitecture

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return a trainable model."""

    @abstractmethod
    def load_weights(self, model: nn.Module) -> dict[str, Any]:
        """Load weights and return auxiliary runtime modules."""

    # might be redundant; can be directly mapped to Sample
    @abstractmethod
    def canonicalize_batch(self, raw_batch: dict[str, Any]) -> DenoiseBatch:
        """Convert upstream inputs into a canonical DenoiseBatch."""

    @abstractmethod
    def prepare_forward_inputs(self, batch: DenoiseBatch, objective_state: dict[str, Any]) -> dict[str, Any]:
        """Prepare model.forward kwargs."""

    @abstractmethod
    def make_parallel_plan(self, parallel_config: ParallelConfig, batch: DenoiseBatch | None = None) -> dict[str, Any]:
        """Build an internal parallel plan for the current model family."""

