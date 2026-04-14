from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
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


class TorchFSDP1ParallelRuntime(ParallelRuntime):
    def __init__(self, mixed_precision: str = "bf16") -> None:
        self.mixed_precision = mixed_precision
        self.model: Any | None = None
        self.optimizer: Any | None = None
        self.use_fsdp = False

    def setup(self) -> None:
        return None

    def parallelize_model(self, model: Any, plan: dict[str, Any]) -> Any:
        del plan
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        if torch.cuda.is_available():
            device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        else:
            device = torch.device("cpu")
        model.to(device)

        if not dist.is_initialized() or world_size == 1:
            self.use_fsdp = False
            self.model = model
            return model

        mp_policy = self._build_mixed_precision()
        self.use_fsdp = True
        self.model = FSDP(
            model,
            device_id=device,
            use_orig_params=True,
            mixed_precision=mp_policy,
            sharding_strategy=ShardingStrategy.FULL_SHARD,
            sync_module_states=True,
        )
        return self.model

    def prepare_optimizer(self, model: Any, optimizer: Any) -> Any:
        del model
        self.optimizer = optimizer
        return optimizer

    def redistribute_batch(self, batch: Any, plan: dict[str, Any]) -> Any:
        del plan
        return batch

    # For FSDP1, we can just call backward and step on the wrapped model and optimizer
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

    def _build_mixed_precision(self) -> Any:
        if self.mixed_precision == "bf16":
            dtype = torch.bfloat16
        elif self.mixed_precision == "fp16":
            dtype = torch.float16
        else:
            dtype = torch.float32
        return MixedPrecision(param_dtype=dtype, reduce_dtype=dtype, buffer_dtype=dtype)

    def is_main_process(self) -> bool:
        if dist is None or not dist.is_available() or not dist.is_initialized():
            return True
        return dist.get_rank() == 0

    def is_distributed(self) -> bool:
        return bool(dist is not None and dist.is_available() and dist.is_initialized())

    def barrier(self) -> None:
        if self.is_distributed():
            dist.barrier()
