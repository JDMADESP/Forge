'''
Test the preprocessing side of the Qwen end-to-end training path.

It does not run Forge training yet.
'''
from __future__ import annotations

import torch
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download
from diffusers import AutoencoderKLQwenImage
from transformers import AutoTokenizer, AutoModel

from image_caption_dataset import ImageCaptionDataset, image_caption_collate
from preprocess_for_qwen import QwenImagePreProcessor


def build_upstream_modules(
    model_name_or_path: str,
    device: str,
    dtype: torch.dtype,
):
    local_repo_dir = snapshot_download(
        repo_id=model_name_or_path,
        allow_patterns=[
            "vae/*",
            "tokenizer/*",
            "text_encoder/*",
        ],
    )

    vae = AutoencoderKLQwenImage.from_pretrained(local_repo_dir, subfolder="vae")
    tokenizer = AutoTokenizer.from_pretrained(local_repo_dir, subfolder="tokenizer")
    text_encoder = AutoModel.from_pretrained(local_repo_dir, subfolder="text_encoder")

    vae = vae.to(device=device, dtype=dtype)
    text_encoder = text_encoder.to(device=device, dtype=dtype)

    return vae, tokenizer, text_encoder


def main():
    dataset = ImageCaptionDataset(
        root="toy_data",
        image_size=256,
        image_folder="images",
        metadata_file="metadata.jsonl",
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=image_caption_collate,
    )

    raw_batch = next(iter(loader))
    print("raw batch keys:", raw_batch.keys())
    print("raw pixel_values shape:", raw_batch["pixel_values"].shape)
    print("raw prompts:", raw_batch["prompts"])
    print("raw file_names:", raw_batch["file_names"])

    device = "cpu"
    dtype = torch.float32
    model_name_or_path = "Qwen/Qwen-Image"

    vae, tokenizer, text_encoder = build_upstream_modules(
        model_name_or_path=model_name_or_path,
        device=device,
        dtype=dtype,
    )

    preprocessor = QwenImagePreProcessor(
        vae=vae,
        tokenizer=tokenizer,
        text_encoder=text_encoder,
        device=device,
        dtype=dtype,
    )

    processed_batch = preprocessor.preprocess_batch(raw_batch)

    print("\nprocessed batch keys:", processed_batch.keys())
    print("latents shape:", processed_batch["latents"].shape)
    print("prompt_embeds shape:", processed_batch["prompt_embeds"].shape)
    print(
        "encoder_hidden_states_mask shape:",
        processed_batch["encoder_hidden_states_mask"].shape
        if processed_batch.get("encoder_hidden_states_mask") is not None
        else None,
    )
    print("img_shapes shape:", processed_batch["img_shapes"].shape)
    print("file_names:", processed_batch["file_names"])
    print("prompts:", processed_batch["prompts"])


if __name__ == "__main__":
    main()