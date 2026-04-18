from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class CheckpointAdapter(ABC):
    @abstractmethod
    def load_config(self, model_name_or_path: str) -> dict[str, Any]:
        """Load the upstream checkpoint config."""

    @abstractmethod
    def build_model_config(self, raw_config: dict[str, Any]) -> Any:
        """Convert the upstream config into an owned model config."""

    @abstractmethod
    def load_state_dict(self, model_name_or_path: str) -> dict[str, torch.Tensor]:
        """Load checkpoint weights from the upstream source."""

    @abstractmethod
    def remap_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Translate upstream parameter names into owned model-core parameter names."""
