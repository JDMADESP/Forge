from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch.nn as nn
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from forge.architectures.sd3 import SD3Architecture
from forge.batch import DenoiseBatch
from forge.parallel.config import ParallelConfig
from forge.parallel.plan import ParallelPlan, StrategySpec
from forge.runtimes.base import ModelRuntime


def load_sd3_transformer_config(model_name_or_path: str) -> dict[str, Any]:
    root = Path(model_name_or_path)
    return json.loads((root / "transformer" / "config.json").read_text(encoding="utf-8"))


class SD3Runtime(ModelRuntime):
    def __init__(self, model_name_or_path: str) -> None:
        transformer_config = load_sd3_transformer_config(model_name_or_path)
        self.model_name_or_path = model_name_or_path
        self.architecture = SD3Architecture(
            model_name_or_path=model_name_or_path,
            transformer_config=transformer_config,
        )
        self.runtime_modules: dict[str, Any] = {}

    def build_model(self) -> nn.Module:
        return self.architecture.build_model()

    def load_weights(self, model: nn.Module) -> dict[str, Any]:
        pretrained = type(model).from_pretrained(self.model_name_or_path, subfolder="transformer")
        model.load_state_dict(pretrained.state_dict())
        del pretrained

        scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
            self.model_name_or_path,
            subfolder="scheduler",
        )
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
        self.runtime_modules = {
            "dit": model,
            "noise_scheduler": scheduler,
        }
        return self.runtime_modules

    def canonicalize_batch(self, raw_batch: dict[str, Any]) -> DenoiseBatch:
        # might need refactor here
        values = {
            "latents": raw_batch.get("latents", raw_batch.get("latent")),
            "prompt_embeds": raw_batch.get("prompt_embeds", raw_batch.get("prompt_embed")),
            "pooled_embeds": raw_batch.get("pooled_embeds", raw_batch.get("pooled_prompt_embed")),
        }
        missing = [
            field
            for field in self.architecture.condition_schema.required_fields
            if values.get(field) is None
        ]
        if missing:
            raise ValueError(f"Raw batch is missing required fields: {missing}")

        sample_ids = raw_batch.get("sample_ids", raw_batch.get("sample_id", []))
        if isinstance(sample_ids, str):
            sample_ids = [sample_ids]

        return DenoiseBatch(
            latents=values["latents"],
            prompt_embeds=values["prompt_embeds"],
            pooled_embeds=values["pooled_embeds"],
            timesteps=raw_batch.get("timesteps"),
            noise=raw_batch.get("noise"),
            attention_mask=raw_batch.get("attention_mask"),
            image_embeds=raw_batch.get("image_embeds", raw_batch.get("image_embed")),
            sample_ids=list(sample_ids),
            metadata={"raw_batch": raw_batch},
        )

    def prepare_forward_inputs(self, batch: DenoiseBatch, objective_state: dict[str, Any]) -> dict[str, Any]:
        return {
            "hidden_states": objective_state["noisy_latents"],
            "timestep": objective_state["timesteps"],
            "encoder_hidden_states": batch.prompt_embeds,
            "pooled_projections": batch.pooled_embeds,
            "return_dict": False,
        }

    def make_parallel_plan(self, parallel_config: ParallelConfig, batch: DenoiseBatch | None = None) -> ParallelPlan:
        del batch
        strategies: list[StrategySpec] = []
        parameter_parallel = parallel_config.parameter_parallel
        if parameter_parallel is not None and parameter_parallel.mode == "fsdp1":
            strategies.append(
                StrategySpec(
                    kind="fsdp1",
                    config={
                        "degree": parameter_parallel.degree,
                        "wrap_block_classes": self.architecture.parallel_spec.wrap_block_classes,
                        "no_shard_modules": self.architecture.parallel_spec.no_shard_modules,
                    },
                )
            )

        sequence_parallel = parallel_config.sequence_parallel
        if sequence_parallel.mode not in {"none", "auto"}:
            raise NotImplementedError("SD3 native/patched sequence parallel is not implemented yet.")

        return ParallelPlan(
            parameter_degree=parameter_parallel.degree if parameter_parallel is not None else 1,
            sequence_degree=sequence_parallel.degree,
            strategy_order=tuple(strategy.kind for strategy in strategies),
            strategies=tuple(strategies),
            required_batch_extras=(),
        )
