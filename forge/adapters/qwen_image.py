from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any

import torch
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from forge.adapters.base import CheckpointAdapter
from forge.model_cores.qwen_image import QwenImageDiTConfig


class QwenImageCheckpointAdapter(CheckpointAdapter):
    def load_config(self, model_name_or_path: str) -> dict[str, Any]:
        root = Path(model_name_or_path)
        return json.loads((root / "transformer" / "config.json").read_text(encoding="utf-8"))

    def build_model_config(self, raw_config: dict[str, Any]) -> QwenImageDiTConfig:
        config_field_names = {field.name for field in fields(QwenImageDiTConfig)}
        config_kwargs = {name: raw_config[name] for name in config_field_names if name in raw_config}
        if "axes_dims_rope" in config_kwargs:
            config_kwargs["axes_dims_rope"] = tuple(config_kwargs["axes_dims_rope"])
        return QwenImageDiTConfig(**config_kwargs)

    def load_state_dict(self, model_name_or_path: str) -> dict[str, torch.Tensor]:
        reference_model = QwenImageTransformer2DModel.from_pretrained(model_name_or_path, subfolder="transformer")
        state_dict = reference_model.state_dict()
        del reference_model
        return state_dict

    def remap_state_dict(self, state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        # Phase A keeps the owned QwenImageDiT module layout aligned with the diffusers reference
        # so we can validate the bridge with a straightforward one-to-one state-dict mapping.
        return dict(state_dict)
