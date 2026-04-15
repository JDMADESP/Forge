from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from diffusers.models._modeling_parallel import ContextParallelConfig
from torch.distributed.checkpoint import load as dcp_load
from torch.distributed.checkpoint import save as dcp_save
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy

from forge.parallel.base import ParallelRuntime
from forge.parallel.plan import ParallelPlan, StrategySpec


class TorchParallelRuntime(ParallelRuntime):
    def __init__(self, mixed_precision: str = "bf16") -> None:
        self.mixed_precision = mixed_precision
        self.model: Any | None = None
        self.optimizer: Any | None = None
        self.use_fsdp = False

    def setup(self, plan: ParallelPlan | None = None) -> None:
        del plan
        return None

    def parallelize_model(self, model: Any, plan: ParallelPlan) -> Any:
        if plan.parameter_degree > 1 and plan.sequence_degree > 1:
            raise NotImplementedError("Combined FSDP + sequence parallel is not implemented yet.")

        device = self._get_device()
        model.to(device)

        sequence_strategy = plan.get_strategy("native_sequence_parallel")
        if sequence_strategy is not None:
            self._apply_native_sequence_parallel(model, sequence_strategy)

        fsdp_strategy = plan.get_strategy("fsdp1")
        if fsdp_strategy is not None:
            model = self._apply_fsdp(model, fsdp_strategy, device)

        self.model = model
        return model

    def prepare_optimizer(self, model: Any, optimizer: Any) -> Any:
        del model
        self.optimizer = optimizer
        return optimizer

    def redistribute_batch(self, batch: Any, plan: ParallelPlan) -> Any:
        for extra in plan.required_batch_extras:
            if extra not in getattr(batch, "model_extras", {}):
                raise ValueError(f"Canonical batch is missing required model extra: {extra}")
        return batch

    def backward(self, loss: Any) -> None:
        loss.backward()

    def step(self, optimizer: Any, scheduler: Any | None = None) -> None:
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()

    def save(self, path: str, state: dict[str, Any]) -> None:
        if self.model is None:
            raise RuntimeError("Model has not been prepared")
        ckpt_dir = Path(path)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        torch.save(state, ckpt_dir / "trainer_meta.pt")
        if not self.use_fsdp:
            torch.save(
                {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict() if self.optimizer is not None else {},
                },
                ckpt_dir / "checkpoint.pt",
            )
            return
        state_dict = {
            "model": get_model_state_dict(
                self.model,
                options=StateDictOptions(full_state_dict=False, cpu_offload=True),
            ),
            "optimizer": get_optimizer_state_dict(
                self.model,
                self.optimizer,
                options=StateDictOptions(full_state_dict=False, cpu_offload=True),
            ) if self.optimizer is not None else {},
        }
        dcp_save(state_dict=state_dict, checkpoint_id=ckpt_dir)

    def load(self, path: str, model: Any, optimizer: Any | None = None) -> dict[str, Any]:
        ckpt_dir = Path(path)
        meta_path = ckpt_dir / "trainer_meta.pt"
        meta = torch.load(meta_path, map_location="cpu") if meta_path.exists() else {}
        if not self.use_fsdp:
            checkpoint = torch.load(ckpt_dir / "checkpoint.pt", map_location="cpu")
            model.load_state_dict(checkpoint["model"])
            if optimizer is not None and checkpoint.get("optimizer"):
                optimizer.load_state_dict(checkpoint["optimizer"])
            return meta
        state_dict = {
            "model": get_model_state_dict(
                model,
                options=StateDictOptions(full_state_dict=False, cpu_offload=True),
            ),
            "optimizer": get_optimizer_state_dict(
                model,
                optimizer,
                options=StateDictOptions(full_state_dict=False, cpu_offload=True),
            ) if optimizer is not None else {},
        }
        dcp_load(state_dict=state_dict, checkpoint_id=ckpt_dir)
        set_model_state_dict(model, model_state_dict=state_dict["model"])
        if optimizer is not None:
            set_optimizer_state_dict(
                model,
                optimizer,
                optim_state_dict=state_dict["optimizer"],
            )
        return meta

    def is_main_process(self) -> bool:
        if not self.is_distributed():
            return True
        return dist.get_rank() == 0

    def is_distributed(self) -> bool:
        return bool(dist.is_available() and dist.is_initialized())

    def barrier(self) -> None:
        if self.is_distributed():
            dist.barrier()

    def _apply_native_sequence_parallel(self, model: Any, strategy: StrategySpec) -> None:
        if not self.is_distributed():
            raise RuntimeError("torch.distributed must be initialized before enabling native sequence parallelism.")

        config = strategy.config
        degree = int(config["degree"])
        if degree != dist.get_world_size():
            raise NotImplementedError(
                "Native sequence parallel currently expects the sequence-parallel degree to match world_size."
            )

        algorithm = str(config["algorithm"])
        cp_kwargs: dict[str, Any] = {"convert_to_fp32": bool(config.get("convert_to_fp32", True))}
        if algorithm == "ulysses":
            cp_kwargs["ulysses_degree"] = degree
        elif algorithm == "ring":
            cp_kwargs["ring_degree"] = degree
        elif algorithm == "ulysses_anything":
            cp_kwargs["ulysses_degree"] = degree
            cp_kwargs["ulysses_anything"] = True
        else:
            raise NotImplementedError(f"Unsupported native sequence-parallel algorithm: {algorithm}")

        model.set_attention_backend(str(config["attention_backend"]))
        model.enable_parallelism(config=ContextParallelConfig(**cp_kwargs))

    def _apply_fsdp(self, model: Any, strategy: StrategySpec, device: torch.device) -> Any:
        world_size = dist.get_world_size() if self.is_distributed() else 1
        degree = int(strategy.config.get("degree", 1))
        if world_size == 1 or degree == 1:
            self.use_fsdp = False
            return model

        if degree != world_size:
            raise NotImplementedError("FSDP currently expects the parameter-parallel degree to match world_size.")

        self.use_fsdp = True
        return FSDP(
            model,
            device_id=device,
            use_orig_params=True,
            mixed_precision=self._build_mixed_precision(),
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            sync_module_states=True,
        )

    def _build_mixed_precision(self) -> Any:
        if self.mixed_precision == "bf16":
            dtype = torch.bfloat16
        elif self.mixed_precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    @staticmethod
    def _get_device() -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        return torch.device("cpu")
