from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ParallelStrategy(ABC):
    @abstractmethod
    def prepare_model(self, model: Any) -> Any:
        """Wrap and return a trainable model."""

    @abstractmethod
    def prepare_optimizer(self, model: Any, optimizer: Any) -> Any:
        """Return an optimizer compatible with the wrapped model."""

    @abstractmethod
    def backward(self, loss: Any) -> None:
        """Run backward for one step."""

    @abstractmethod
    def step(self, optimizer: Any, scheduler: Any | None = None) -> None:
        """Run the optimizer update and gradient clear."""

    @abstractmethod
    def save(self, path: str, state: dict[str, Any]) -> None:
        """Save strategy-managed state."""

    @abstractmethod
    def load(self, path: str, model: Any, optimizer: Any | None = None) -> dict[str, Any]:
        """Load strategy-managed state and return trainer metadata."""

    def clip_grad_norm_(self, model: Any, max_norm: float) -> None:
        del model, max_norm

    def apply_plan(self, model: Any, plan: Any | None = None) -> Any:
        del plan
        return model

    def redistribute(self, tensor: Any, layout: Any | None = None) -> Any:
        del layout
        return tensor
