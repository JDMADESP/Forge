from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from forge.state import BatchState


class ModelAdapter(ABC):
    @abstractmethod
    # used to load and prepare the model and any auxiliary modules (e.g. text encoder, VAE) for training
    def build_modules(self, args: Any) -> dict[str, Any]:
        """Load and return trainable and auxiliary modules."""

    # used to convert raw dataloader output into a shared batch state for the training pipeline
    @abstractmethod
    def prepare_batch(self, raw_batch: dict[str, Any]) -> BatchState:
        """Convert dataloader output into a shared batch state."""

    @abstractmethod
    def forward_loss(self, batch_state: BatchState) -> dict[str, Any]:
        """Run one training forward pass and return loss plus metadata."""

    @abstractmethod
    def validation_generate(self, prompts: list[str], output_dir: str) -> dict[str, Any]:
        """Generate validation artifacts for a fixed prompt set."""

    # used after a model is wrapped by the strategy like FSDP
    def set_train_model(self, model: Any) -> None:
        """Bind the wrapped trainable model back to the adapter when needed."""
        # del model to avoid unused argument warning in some adapters that don't need this method
        del model
