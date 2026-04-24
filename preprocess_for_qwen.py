from __future__ import annotations
from typing import Any
import torch

class QwenImagePreProcessor:
    def __init__(
        self,
        vae,
        tokenizer,
        text_encoder,
        device : str = "cuda",
        dtype :  torch.dtype = torch.bfloat16,
    ):
        self.vae = vae.to(device)
        self.tokenizer = tokenizer
        self.text_encoder = text_encoder.to(device)
        self.device = device
        self.dtype = dtype

        # place models into evaluation mode
        self.vae.eval()
        self.text_encoder.eval()

    @torch.no_grad()
    def preprocess_batch(self, batch):
        pixel_values = batch["pixel_values"].to(self.device, dtype=self.dtype)
        prompts = batch["prompts"]
        if pixel_values.ndim == 4:
            pixel_values = pixel_values.unsqueeze(2)

        # first need to encode images into latent vectors
        latent_dist = self.vae.encode(pixel_values).latent_dist # encoded output of image as probability distribution
        latents = latent_dist.sample()

        if hasattr(self.vae, "config") and hasattr(self.vae.config, "scaling_factor"):
            latents = latents * self.vae.config.scaling_factor

        if latents.ndim == 5:
            if latents.shape[2] == 1:
                latents = latents.squeeze(2)
        elif latents.shape[1] == 1:
            latents = latents.squeeze(1)

        # now need to tokenize prompt
        tokenized = self.tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
        input_ids = tokenized["input_ids"].to(self.device)
        encoder_hidden_states_mask = tokenized.get("attention_mask")
        if encoder_hidden_states_mask is not None:
            encoder_hidden_states_mask = encoder_hidden_states_mask.to(self.device)

        # text to embeddings
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=encoder_hidden_states_mask,
        )

        prompt_embeds = text_outputs.last_hidden_state

        batch_size, _, latent_h, latent_w = latents.shape
        img_shapes = []
        for _ in range(batch_size):
            img_shapes.append([latent_h, latent_w])
        img_shapes = torch.tensor(img_shapes, device=self.device, dtype=torch.long,)

        return {
            "latents": latents,
            "prompt_embeds": prompt_embeds,
            "encoder_hidden_states_mask": encoder_hidden_states_mask,
            "img_shapes" : img_shapes,
            "file_names": batch["file_names"],
            "prompts": prompts,
        }

    def convert_to_denoise_batch(self, processed_batch):
        '''
        Using the following class as reference:
            class DenoiseBatch:
                """Canonical denoising-step inputs consumed inside Forge."""

                latents: torch.Tensor
                prompt_embeds: torch.Tensor
                pooled_embeds: torch.Tensor | None = None
                timesteps: torch.Tensor | None = None
                noise: torch.Tensor | None = None
                attention_mask: torch.Tensor | None = None
                image_embeds: torch.Tensor | None = None
                sample_ids: list[str] = field(default_factory=list)
                metadata: dict[str, Any] = field(default_factory=dict)
                model_extras: dict[str, Any] = field(default_factory=dict)
        
        '''
        model_extras = dict()
        if processed_batch.get("encoder_hidden_states_mask") is not None:
            model_extras["encoder_hidden_states_mask"] = processed_batch["encoder_hidden_states_mask"]

        if processed_batch.get("img_shapes") is not None:
            model_extras["img_shapes"] = processed_batch["img_shapes"]
        
        return DenoiseBatch(
                latents=processed_batch["latents"],
                prompt_embeds=processed_batch["prompt_embeds"],
                sample_ids=processed_batch["file_names"],
                metadata={"prompts":processed_batch["prompts"]},
                model_extras=model_extras
               )