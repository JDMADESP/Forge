from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch

from forge.batch import DenoiseBatch
from forge.runtimes.base import ModelRuntime


class TrainingObjective(ABC):
    """Training objective semantics."""

    @abstractmethod
    def prepare(self, batch: DenoiseBatch, runtime: ModelRuntime, model: Any) -> dict[str, Any]:
        """Prepare objective state for one step."""

    @abstractmethod
    def compute_loss(
        self,
        model_outputs: Any,
        batch: DenoiseBatch,
        objective_state: dict[str, Any],
    ) -> tuple[torch.Tensor, dict[str, float], dict[str, Any]]:
        """Compute scalar loss plus metrics and artifacts."""

