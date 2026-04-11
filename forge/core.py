from __future__ import annotations

from dataclasses import dataclass

from forge.objectives.base import TrainingObjective
from forge.objectives.flow_match import FlowMatchObjective
from forge.parallel.base import ParallelRuntime
from forge.parallel.config import ParallelConfig
from forge.parallel.fsdp1 import TorchFSDP1ParallelRuntime
from forge.runtimes.base import ModelRuntime
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
    if model_family != "sd3":
        raise ValueError(f"Unsupported model family: {model_family}")

    model_runtime = SD3Runtime(model_name_or_path=model_name_or_path)
    objective = FlowMatchObjective()

    if parallel_config.backend != "torch":
        raise ValueError(f"Unsupported parallel backend: {parallel_config.backend}")
    if parallel_config.dp_mode != "fsdp1":
        raise ValueError(f"Unsupported dp mode for current implementation: {parallel_config.dp_mode}")
    parallel_runtime = TorchFSDP1ParallelRuntime(mixed_precision=parallel_config.mixed_precision)
    return ForgeCore(
        model_runtime=model_runtime,
        objective=objective,
        parallel_runtime=parallel_runtime,
    )
