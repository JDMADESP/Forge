from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F

from forge.adapters.base import ModelAdapter
from forge.state import BatchState

try:
    from diffusers import StableDiffusion3Pipeline
except Exception as exc:  # pragma: no cover - import guard for local environments
    StableDiffusion3Pipeline = None
    _DIFFUSERS_IMPORT_ERROR = exc
else:
    _DIFFUSERS_IMPORT_ERROR = None


class DiffusersSD3Adapter(ModelAdapter):
    def __init__(self) -> None:
        self.pipeline: Any | None = None
        self.train_model: Any | None = None
        self.runtime_modules: dict[str, Any] = {}
        self.trainable_modules: list[str] = []
        self.primary_train_model: Any | None = None

    def build_modules(self, args: Any) -> dict[str, Any]:
        if StableDiffusion3Pipeline is None:
            raise RuntimeError("diffusers is unavailable in this environment") from _DIFFUSERS_IMPORT_ERROR

        pipe = StableDiffusion3Pipeline.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float32,
        )
        pipe.scheduler.set_timesteps(pipe.scheduler.config.num_train_timesteps)
        self.pipeline = pipe
        #register all submodules needed, including trainable and non-trainable ones
        self.runtime_modules = {
            "dit": pipe.transformer,
            "vae": pipe.vae,
            "noise_scheduler": pipe.scheduler,
            "text_encoder": pipe.text_encoder,
            "text_encoder_2": pipe.text_encoder_2,
            "text_encoder_3": pipe.text_encoder_3,
            "tokenizer": pipe.tokenizer,
            "tokenizer_2": pipe.tokenizer_2,
            "tokenizer_3": pipe.tokenizer_3,
        }

        #you would need to unfreeze more modules for a real training run
        self.trainable_modules = ["dit"]
        self.primary_train_model = self.runtime_modules["dit"]

        self.runtime_modules["vae"].requires_grad_(False)
        self.runtime_modules["text_encoder"].requires_grad_(False)
        if self.runtime_modules["text_encoder_2"] is not None:
            self.runtime_modules["text_encoder_2"].requires_grad_(False)
        if self.runtime_modules["text_encoder_3"] is not None:
            self.runtime_modules["text_encoder_3"].requires_grad_(False)

        return {
            "model": self.primary_train_model,
            "runtime_modules": self.runtime_modules,
            "trainable_modules": self.trainable_modules,
        }

    def prepare_batch(self, raw_batch: dict[str, Any]) -> BatchState:
        return BatchState(raw_batch=raw_batch)

    def set_train_model(self, model: Any) -> None:
        self.train_model = model
        self.primary_train_model = model
        self.runtime_modules["dit"] = model
        device = next(model.parameters()).device
        for module_name in ("vae", "text_encoder", "text_encoder_2", "text_encoder_3"):
            module = self.runtime_modules.get(module_name)
            if module is not None:
                module.to(device)
        if self.pipeline is not None:
            self.pipeline.transformer = model
            self.pipeline.vae = self.runtime_modules["vae"]
            self.pipeline.text_encoder = self.runtime_modules["text_encoder"]
            self.pipeline.text_encoder_2 = self.runtime_modules["text_encoder_2"]
            self.pipeline.text_encoder_3 = self.runtime_modules["text_encoder_3"]

    def forward_loss(self, batch_state: BatchState) -> dict[str, Any]:
        if self.train_model is None and self.primary_train_model is None:
            raise RuntimeError("build_modules must be called before forward_loss")
        pixel_values = batch_state.raw_batch["pixel_values"]
        prompts = batch_state.raw_batch["prompts"]
        model = self.train_model if self.train_model is not None else self.primary_train_model

        first_param = next(model.parameters())
        device = first_param.device
        dtype = first_param.dtype
        scheduler = self.runtime_modules["noise_scheduler"]
        latents = self._encode_latents(pixel_values, device)
        prompt_embeds, pooled_prompt_embeds = self._encode_prompts(prompts, device)
        latents = latents.to(dtype=dtype)
        prompt_embeds = prompt_embeds.to(device=device, dtype=dtype)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device=device, dtype=dtype)

        noise = torch.randn_like(latents)
        bsz = latents.shape[0]
        step_indices = torch.randint(
            0,
            scheduler.timesteps.shape[0],
            (bsz,),
            device=device,
            dtype=torch.long,
        )
        timesteps = scheduler.timesteps.to(device=device)[step_indices]
        sigmas = scheduler.sigmas.to(device=device, dtype=latents.dtype)[step_indices]
        sigmas = sigmas.view(bsz, *([1] * (latents.ndim - 1)))

        noisy_latents = (1.0 - sigmas) * latents + sigmas * noise
        target = noise - latents

        model_pred = model(
            hidden_states=noisy_latents,
            timestep=timesteps,
            encoder_hidden_states=prompt_embeds,
            pooled_projections=pooled_prompt_embeds,
            return_dict=False,
        )[0]
        loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
        batch_state.loss_dict["loss"] = loss
        batch_state.metrics["loss"] = float(loss.detach().item())
        return {"loss": loss, "metrics": batch_state.metrics, "artifacts": batch_state.artifacts}

    def validation_generate(self, prompts: list[str], output_dir: str) -> dict[str, Any]:
        if self.pipeline is None or self.primary_train_model is None:
            raise RuntimeError("build_modules must be called before validation_generate")
        if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
            return {"sample_paths": []}

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        transformer = self.pipeline.transformer
        was_training = transformer.training
        transformer.eval()
        maybe_fsdp_model = self.train_model
        full_param_context = nullcontext()
        if maybe_fsdp_model is not None and hasattr(type(maybe_fsdp_model), "summon_full_params"):
            full_param_context = type(maybe_fsdp_model).summon_full_params(
                maybe_fsdp_model, recurse=True
            )

        images: list[str] = []
        with full_param_context, torch.no_grad():
            for index, prompt in enumerate(prompts):
                image = self.pipeline(
                    prompt=prompt,
                    num_inference_steps=20,
                    guidance_scale=4.5,
                ).images[0]
                image_path = output_path / f"sample_{index:03d}.png"
                image.save(image_path)
                images.append(str(image_path))

        if was_training:
            transformer.train()
        return {"sample_paths": images}

    def _encode_latents(self, pixel_values: torch.Tensor, device: torch.device) -> torch.Tensor:
        vae = self.runtime_modules["vae"]
        pixel_values = pixel_values.to(device=device, dtype=vae.dtype)
        with torch.no_grad():
            latents = vae.encode(pixel_values).latent_dist.sample()
            latents = (latents - vae.config.shift_factor) * vae.config.scaling_factor
        return latents

    def _encode_prompts(self, prompts: list[str], device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        tokenizer = self.runtime_modules["tokenizer"]
        tokenizer_2 = self.runtime_modules["tokenizer_2"]
        tokenizer_3 = self.runtime_modules["tokenizer_3"]
        text_encoder = self.runtime_modules["text_encoder"]
        text_encoder_2 = self.runtime_modules["text_encoder_2"]
        text_encoder_3 = self.runtime_modules["text_encoder_3"]

        if (
            tokenizer is None
            or tokenizer_2 is None
            or tokenizer_3 is None
            or text_encoder is None
            or text_encoder_2 is None
            or text_encoder_3 is None
        ):
            raise RuntimeError("SD3 prompt encoders/tokenizers must all be available")

        with torch.no_grad():
            clip_prompt_embeds_list: list[torch.Tensor] = []
            pooled_prompt_embeds_list: list[torch.Tensor] = []

            for current_tokenizer, current_encoder in (
                (tokenizer, text_encoder),
                (tokenizer_2, text_encoder_2),
            ):
                text_inputs = current_tokenizer(
                    prompts,
                    padding="max_length",
                    max_length=current_tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                input_ids = text_inputs.input_ids.to(device)
                outputs = current_encoder(input_ids, output_hidden_states=True)
                pooled_prompt_embeds_list.append(outputs[0])
                clip_prompt_embeds_list.append(outputs.hidden_states[-2])

            clip_prompt_embeds = torch.cat(clip_prompt_embeds_list, dim=-1)

            max_sequence_length = 256
            text_inputs = tokenizer_3(
                prompts,
                padding="max_length",
                max_length=max_sequence_length,
                truncation=True,
                add_special_tokens=True,
                return_tensors="pt",
            )
            t5_input_ids = text_inputs.input_ids.to(device)
            t5_prompt_embeds = text_encoder_3(t5_input_ids)[0]

            clip_prompt_embeds = torch.nn.functional.pad(
                clip_prompt_embeds,
                (0, t5_prompt_embeds.shape[-1] - clip_prompt_embeds.shape[-1]),
            )
            prompt_embeds = torch.cat([clip_prompt_embeds, t5_prompt_embeds], dim=-2)
            pooled_prompt_embeds = torch.cat(pooled_prompt_embeds_list, dim=-1)

        return prompt_embeds, pooled_prompt_embeds
