from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from forge.adapters.base import ModelAdapter
from forge.state import BatchState

if not hasattr(torch, "xpu"):
    class _XpuStub:
        @staticmethod
        def empty_cache() -> None:
            return None

        @staticmethod
        def device_count() -> int:
            return 0

        @staticmethod
        def manual_seed(seed: int) -> None:
            del seed

        @staticmethod
        def manual_seed_all(seed: int) -> None:
            del seed

        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def _is_in_bad_fork() -> bool:
            return False

    torch.xpu = _XpuStub()  # type: ignore[attr-defined]

if not hasattr(torch, "library"):
    class _LibraryStub:
        @staticmethod
        def register_fake(*args: Any, **kwargs: Any):
            del args, kwargs

            def decorator(fn: Any) -> Any:
                return fn

            return decorator

    torch.library = _LibraryStub()  # type: ignore[attr-defined]
elif not hasattr(torch.library, "register_fake"):
    def _register_fake(*args: Any, **kwargs: Any):
        del args, kwargs

        def decorator(fn: Any) -> Any:
            return fn

        return decorator

    torch.library.register_fake = _register_fake  # type: ignore[attr-defined]

try:
    from diffusers import FlowMatchEulerDiscreteScheduler, StableDiffusion3Pipeline
except Exception as exc:  # pragma: no cover
    FlowMatchEulerDiscreteScheduler = None
    StableDiffusion3Pipeline = None
    _DIFFUSERS_IMPORT_ERROR = exc
else:
    _DIFFUSERS_IMPORT_ERROR = None


class SD3DiTAdapter(ModelAdapter):
    def __init__(self) -> None:
        self.train_model: Any | None = None
        self.primary_train_model: Any | None = None
        self.runtime_modules: dict[str, Any] = {}
        self.trainable_modules: list[str] = []

    def build_modules(self, args: Any) -> dict[str, Any]:
        if StableDiffusion3Pipeline is None or FlowMatchEulerDiscreteScheduler is None:
            raise RuntimeError("diffusers is unavailable in this environment") from _DIFFUSERS_IMPORT_ERROR

        pipe = StableDiffusion3Pipeline.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32)
        transformer = pipe.transformer
        scheduler = pipe.scheduler
        scheduler.set_timesteps(scheduler.config.num_train_timesteps)
        del pipe

        self.runtime_modules = {
            "dit": transformer,
            "noise_scheduler": scheduler,
        }
        self.trainable_modules = ["dit"]
        self.primary_train_model = transformer
        return {
            "model": transformer,
            "runtime_modules": self.runtime_modules,
            "trainable_modules": self.trainable_modules,
        }

    def prepare_batch(self, raw_batch: dict[str, Any]) -> BatchState:
        required = ("latent", "prompt_embed", "pooled_prompt_embed")
        missing = [key for key in required if key not in raw_batch]
        if missing:
            raise ValueError(f"Raw batch is missing required fields: {missing}")

        batch_state = BatchState(raw_batch=raw_batch)
        batch_state.model_inputs["latent"] = raw_batch["latent"]
        batch_state.model_inputs["prompt_embed"] = raw_batch["prompt_embed"]
        batch_state.model_inputs["pooled_prompt_embed"] = raw_batch["pooled_prompt_embed"]

        for optional_key in ("timesteps", "noise", "attention_mask", "position_embedding", "image_embed"):
            if optional_key in raw_batch:
                batch_state.model_inputs[optional_key] = raw_batch[optional_key]
        return batch_state

    def set_train_model(self, model: Any) -> None:
        self.train_model = model
        self.primary_train_model = model
        self.runtime_modules["dit"] = model

    def forward_loss(self, batch_state: BatchState) -> dict[str, Any]:
        if self.primary_train_model is None and self.train_model is None:
            raise RuntimeError("build_modules must be called before forward_loss")

        model = self.train_model if self.train_model is not None else self.primary_train_model
        first_param = next(model.parameters())
        device = first_param.device
        dtype = first_param.dtype

        scheduler = self.runtime_modules["noise_scheduler"]
        latent = batch_state.model_inputs["latent"].to(device=device, dtype=dtype)
        prompt_embed = batch_state.model_inputs["prompt_embed"].to(device=device, dtype=dtype)
        pooled_prompt_embed = batch_state.model_inputs["pooled_prompt_embed"].to(device=device, dtype=dtype)

        batch_size = latent.shape[0]
        noise = batch_state.model_inputs.get("noise")
        if noise is None:
            noise = torch.randn_like(latent)
        else:
            noise = noise.to(device=device, dtype=dtype)

        raw_timesteps = batch_state.model_inputs.get("timesteps")
        if raw_timesteps is None:
            step_indices = torch.randint(
                0,
                scheduler.timesteps.shape[0],
                (batch_size,),
                device=device,
                dtype=torch.long,
            )
            timesteps = scheduler.timesteps.to(device=device)[step_indices]
        else:
            timesteps, step_indices = self._resolve_timesteps(raw_timesteps.view(batch_size), scheduler, device)

        sigmas = scheduler.sigmas.to(device=device, dtype=dtype)[step_indices]
        sigmas = sigmas.view(batch_size, *([1] * (latent.ndim - 1)))

        noisy_latent = (1.0 - sigmas) * latent + sigmas * noise
        target = noise - latent

        model_pred = model(
            hidden_states=noisy_latent,
            timestep=timesteps,
            encoder_hidden_states=prompt_embed,
            pooled_projections=pooled_prompt_embed,
            return_dict=False,
        )[0]
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

        batch_state.targets["target"] = target.detach()
        batch_state.loss_dict["loss"] = loss
        batch_state.metrics["loss"] = float(loss.detach().item())
        batch_state.artifacts["sample_ids"] = batch_state.raw_batch.get("sample_id", [])
        return {"loss": loss, "metrics": batch_state.metrics, "artifacts": batch_state.artifacts}

    def validation_generate(
        self,
        prompts: list[str],
        output_dir: str,
        seed: int | None = None,
        generator: Any | None = None,
    ) -> dict[str, Any]:
        del prompts, seed, generator
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        summary_path = output_path / "validation_skipped.txt"
        summary_path.write_text("Validation generate is intentionally disabled for DiT-only M0.\n", encoding="utf-8")
        return {"sample_paths": [str(summary_path)]}

    @staticmethod
    def _resolve_timesteps(
        timesteps: torch.Tensor,
        scheduler: Any,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scheduler_timesteps = scheduler.timesteps.to(device=device)
        raw_timesteps = timesteps.to(device=device)
        indices: list[int] = []
        resolved_timesteps: list[torch.Tensor] = []
        max_index = scheduler_timesteps.shape[0] - 1
        for timestep in raw_timesteps:
            matches = (scheduler_timesteps == timestep).nonzero(as_tuple=False)
            if matches.numel() > 0:
                index = int(matches[0].item())
                indices.append(index)
                resolved_timesteps.append(scheduler_timesteps[index])
                continue

            maybe_index = int(timestep.item())
            if 0 <= maybe_index <= max_index:
                indices.append(maybe_index)
                resolved_timesteps.append(scheduler_timesteps[maybe_index])
                continue

            raise ValueError(f"Provided timestep {float(timestep.item())} is not present in scheduler.timesteps")

        return (
            torch.stack(resolved_timesteps).to(device=device, dtype=scheduler_timesteps.dtype),
            torch.tensor(indices, device=device, dtype=torch.long),
        )
