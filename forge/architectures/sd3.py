from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn

try:
    from diffusers.models.transformers.transformer_sd3 import SD3Transformer2DModel
except Exception as exc:  # pragma: no cover
    SD3Transformer2DModel = None
    _DIFFUSERS_IMPORT_ERROR = exc
else:
    _DIFFUSERS_IMPORT_ERROR = None

from forge.architectures.base import (
    ArchitectureParallelSpec,
    ConditionSchema,
    ModelArchitecture,
)


@dataclass
class SD3Architecture(ModelArchitecture):
    model_name_or_path: str
    transformer_config: dict[str, Any]

    def __post_init__(self) -> None:
        self.condition_schema = ConditionSchema(
            required_fields=("latents", "prompt_embeds", "pooled_embeds"),
            optional_fields=("timesteps", "noise", "attention_mask", "image_embeds"),
            forward_arg_map={
                "latents": "hidden_states",
                "prompt_embeds": "encoder_hidden_states",
                "pooled_embeds": "pooled_projections",
                "timesteps": "timestep",
            },
        )
        self.parallel_spec = ArchitectureParallelSpec(
            wrap_block_classes=("JointTransformerBlock",),
            no_shard_modules=("pos_embed", "time_text_embed"),
            shardable_inputs={"prompt_embeds": 1},
            replicate_inputs=("pooled_embeds", "timesteps"),
        )

    def build_model(self) -> nn.Module:
        if SD3Transformer2DModel is None:
            raise RuntimeError("diffusers SD3 transformer is unavailable") from _DIFFUSERS_IMPORT_ERROR
        return SD3Transformer2DModel.from_config(self.transformer_config)
