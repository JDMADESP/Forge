from __future__ import annotations

from dataclasses import dataclass

from forge.objectives.base import TrainingObjective
from forge.objectives.flow_match import FlowMatchObjective
from forge.parallel.base import ParallelRuntime
from forge.parallel.config import ParallelConfig
from forge.parallel.torch_runtime import TorchParallelRuntime
from forge.runtimes.base import ModelRuntime
from forge.runtimes.qwen_image import QwenImageRuntime
from forge.runtimes.sd3 import SD3Runtime


@dataclass
class ForgeCore:
    model_runtime: ModelRuntime
    objective: TrainingObjective
    parallel_runtime: ParallelRuntime


def create_core(
    model_family: str,
    model_name_or_path: str,
    parallel_config: ParallelConfig,
) -> ForgeCore:
    registry: dict[str, tuple[type[ModelRuntime], type[TrainingObjective]]] = {
        "sd3": (SD3Runtime, FlowMatchObjective),
        "qwen_image": (QwenImageRuntime, FlowMatchObjective),
    }
    if model_family not in registry:
        raise ValueError(f"Unsupported model family: {model_family}")

    runtime_cls, objective_cls = registry[model_family]
    model_runtime = runtime_cls(model_name_or_path=model_name_or_path)
    objective = objective_cls()

    if parallel_config.backend != "torch":
        raise ValueError(f"Unsupported parallel backend: {parallel_config.backend}")
    parallel_runtime = TorchParallelRuntime(mixed_precision=parallel_config.mixed_precision)
    return ForgeCore(
        model_runtime=model_runtime,
        objective=objective,
        parallel_runtime=parallel_runtime,
    )
