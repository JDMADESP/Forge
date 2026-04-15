from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch.nn as nn
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler

from forge.architectures.qwen_image import QwenImageArchitecture
from forge.batch import DenoiseBatch
from forge.parallel.config import ParallelConfig
from forge.parallel.plan import ParallelPlan, StrategySpec
from forge.runtimes.base import ModelRuntime


def load_qwen_image_transformer_config(model_name_or_path: str) -> dict[str, Any]:
    root = Path(model_name_or_path)
    return json.loads((root / "transformer" / "config.json").read_text(encoding="utf-8"))


class QwenImageRuntime(ModelRuntime):
    def __init__(self, model_name_or_path: str) -> None:
        transformer_config = load_qwen_image_transformer_config(model_name_or_path)
        self.model_name_or_path = model_name_or_path
        self.architecture = QwenImageArchitecture(
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
        values = {
            "latents": raw_batch.get("latents", raw_batch.get("latent", raw_batch.get("hidden_states"))),
            "prompt_embeds": raw_batch.get(
                "prompt_embeds",
                raw_batch.get("prompt_embed", raw_batch.get("encoder_hidden_states")),
            ),
        }
        missing = [field for field in self.architecture.condition_schema.required_fields if values.get(field) is None]
        if missing:
            raise ValueError(f"Raw batch is missing required fields: {missing}")

        model_extras = {
            "encoder_hidden_states_mask": raw_batch.get(
                "encoder_hidden_states_mask",
                raw_batch.get("attention_mask"),
            ),
            "img_shapes": raw_batch.get("img_shapes"),
        }
        if raw_batch.get("guidance") is not None:
            model_extras["guidance"] = raw_batch["guidance"]
        if raw_batch.get("attention_kwargs") is not None:
            model_extras["attention_kwargs"] = raw_batch["attention_kwargs"]

        required_extras = ("encoder_hidden_states_mask", "img_shapes")
        missing_extras = [name for name in required_extras if model_extras.get(name) is None]
        if missing_extras:
            raise ValueError(f"Raw batch is missing required QwenImage extras: {missing_extras}")

        sample_ids = raw_batch.get("sample_ids", raw_batch.get("sample_id", []))
        if isinstance(sample_ids, str):
            sample_ids = [sample_ids]

        return DenoiseBatch(
            latents=values["latents"],
            prompt_embeds=values["prompt_embeds"],
            timesteps=raw_batch.get("timesteps"),
            noise=raw_batch.get("noise"),
            sample_ids=list(sample_ids),
            metadata={"raw_batch": raw_batch},
            model_extras=model_extras,
        )

    def prepare_forward_inputs(self, batch: DenoiseBatch, objective_state: dict[str, Any]) -> dict[str, Any]:
        model_extras = batch.model_extras
        return {
            "hidden_states": objective_state["noisy_latents"],
            "encoder_hidden_states": batch.prompt_embeds,
            "encoder_hidden_states_mask": model_extras["encoder_hidden_states_mask"],
            "timestep": objective_state["timesteps"],
            "img_shapes": model_extras["img_shapes"],
            "guidance": model_extras.get("guidance"),
            "attention_kwargs": model_extras.get("attention_kwargs"),
            "return_dict": False,
        }

    def make_parallel_plan(self, parallel_config: ParallelConfig, batch: DenoiseBatch | None = None) -> ParallelPlan:
        del batch
        spec = self.architecture.parallel_spec
        strategies: list[StrategySpec] = []
        required_batch_extras: set[str] = set()

        parameter_parallel = parallel_config.parameter_parallel
        if parameter_parallel is not None and parameter_parallel.mode == "fsdp1":
            strategies.append(
                StrategySpec(
                    kind="fsdp1",
                    config={
                        "degree": parameter_parallel.degree,
                        "wrap_block_classes": spec.wrap_block_classes,
                        "no_shard_modules": spec.no_shard_modules,
                    },
                )
            )

        sequence_parallel = parallel_config.sequence_parallel
        if sequence_parallel.mode in {"auto", "native"}:
            native_spec = spec.native_sequence_parallel
            if native_spec is None:
                raise ValueError("QwenImage architecture does not declare native sequence parallel support.")
            if sequence_parallel.degree > 1:
                algorithm = sequence_parallel.algorithm or native_spec.default_algorithm
                if algorithm not in native_spec.supported_algorithms:
                    raise ValueError(
                        f"Unsupported native sequence-parallel algorithm '{algorithm}'. "
                        f"Supported: {native_spec.supported_algorithms}"
                    )
                strategies.insert(
                    0,
                    StrategySpec(
                        kind="native_sequence_parallel",
                        config={
                            "degree": sequence_parallel.degree,
                            "algorithm": algorithm,
                            "attention_backend": sequence_parallel.attention_backend,
                            "convert_to_fp32": sequence_parallel.convert_to_fp32,
                            "ulysses_anything": sequence_parallel.ulysses_anything,
                        },
                    ),
                )
                required_batch_extras.update(native_spec.required_batch_extras)
        elif sequence_parallel.mode == "patched":
            raise NotImplementedError("QwenImage patched sequence parallel is not implemented.")

        return ParallelPlan(
            parameter_degree=parameter_parallel.degree if parameter_parallel is not None else 1,
            sequence_degree=sequence_parallel.degree,
            strategy_order=tuple(strategy.kind for strategy in strategies),
            strategies=tuple(strategies),
            required_batch_extras=tuple(sorted(required_batch_extras)),
        )
