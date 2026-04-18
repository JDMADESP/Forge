from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from diffusers.models.modeling_utils import ModelMixin


class BaseDiT(ModelMixin, ABC):
    """Minimal owned diffusion-transformer contract."""

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        """Run the family-specific DiT forward pass."""
