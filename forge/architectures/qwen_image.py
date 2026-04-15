from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch.nn as nn
from diffusers.models.transformers.transformer_qwenimage import QwenImageTransformer2DModel

from forge.architectures.base import (
    ArchitectureParallelSpec,
    ConditionSchema,
    ModelArchitecture,
    NativeSequenceParallelSpec,
)


@dataclass
class QwenImageArchitecture(ModelArchitecture):
    model_name_or_path: str
    transformer_config: dict[str, Any]

    def __post_init__(self) -> None:
        self.condition_schema = ConditionSchema(
            required_fields=("latents", "prompt_embeds"),
            optional_fields=(
                "timesteps",
                "noise",
                "encoder_hidden_states_mask",
                "img_shapes",
                "guidance",
                "attention_kwargs",
            ),
            forward_arg_map={
                "latents": "hidden_states",
                "prompt_embeds": "encoder_hidden_states",
                "timesteps": "timestep",
            },
        )
        self.parallel_spec = ArchitectureParallelSpec(
            wrap_block_classes=("QwenImageTransformerBlock",),
            no_shard_modules=("pos_embed", "time_text_embed"),
            native_sequence_parallel=NativeSequenceParallelSpec(
                supported_algorithms=("ulysses", "ring", "ulysses_anything"),
                default_algorithm="ulysses",
                required_batch_extras=("encoder_hidden_states_mask", "img_shapes"),
            ),
        )

    def build_model(self) -> nn.Module:
        return QwenImageTransformer2DModel.from_config(self.transformer_config)
