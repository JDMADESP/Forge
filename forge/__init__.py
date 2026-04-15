"""Forge diffusion training engine package."""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[1]
_VENDORED_DIFFUSERS_SRC = _ROOT / "diffusers" / "src"
if _VENDORED_DIFFUSERS_SRC.exists():
    vendored_path = str(_VENDORED_DIFFUSERS_SRC)
    if vendored_path not in sys.path:
        sys.path.insert(0, vendored_path)

try:
    import transformers

    if not hasattr(transformers, "AutoImageProcessor"):
        class _AutoImageProcessorStub:
            @classmethod
            def from_pretrained(cls, *args, **kwargs):
                raise NotImplementedError("AutoImageProcessor is unavailable in this transformers build.")

        transformers.AutoImageProcessor = _AutoImageProcessorStub
except Exception:  # noqa: BLE001
    pass

from forge.batch import DenoiseBatch
from forge.core import ForgeCore, create_core
from forge.trainer import Trainer
from forge.training_args import TrainingEngineArgs

__all__ = ["DenoiseBatch", "ForgeCore", "Trainer", "TrainingEngineArgs", "create_core"]
