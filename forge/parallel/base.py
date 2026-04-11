from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ParallelRuntime(ABC):
    @abstractmethod
    def setup(self) -> None:
        """Initialize runtime-local distributed state if needed."""

    @abstractmethod
    def parallelize_model(self, model: Any, plan: dict[str, Any]) -> Any:
        """Wrap and return a trainable model."""

    @abstractmethod
    def prepare_optimizer(self, model: Any, optimizer: Any) -> Any:
        """Return an optimizer compatible with the wrapped model."""

    @abstractmethod
    def redistribute_batch(self, batch: Any, plan: dict[str, Any]) -> Any:
        """Redistribute a canonical batch according to the internal plan."""

    @abstractmethod
    def backward(self, loss: Any) -> None:
        """Run backward for one step."""

    @abstractmethod
    def step(self, optimizer: Any, scheduler: Any | None = None) -> None:
        """Run the optimizer update and gradient clear."""

    @abstractmethod
    def save(self, path: str, state: dict[str, Any]) -> None:
        """Save runtime-managed state."""

    @abstractmethod
    def load(self, path: str, model: Any, optimizer: Any | None = None) -> dict[str, Any]:
        """Load runtime-managed state and return trainer metadata."""

    def clip_grad_norm_(self, model: Any, max_norm: float) -> None:
        del model, max_norm

    def is_main_process(self) -> bool:
        return True

    def is_distributed(self) -> bool:
        return False

    def barrier(self) -> None:
        return None
