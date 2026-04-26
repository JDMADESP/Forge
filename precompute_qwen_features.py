'''
This file precomputes the tensors, in order to save memory
'''
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from huggingface_hub import snapshot_download
from diffusers import AutoencoderKLQwenImage
from transformers import AutoTokenizer, AutoModel

from image_caption_dataset import ImageCaptionDataset, image_caption_collate
from preprocess_for_qwen import QwenImagePreProcessor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Precompute Qwen training features")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_data_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--image_folder", default="images")
    parser.add_argument("--metadata_file", default="metadata.jsonl")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--mixed_precision", default="fp32")
    return parser.parse_args()


def choose_device_and_dtype(mixed_precision: str) -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        if mixed_precision == "bf16":
            return "cuda", torch.bfloat16
        if mixed_precision == "fp16":
            return "cuda", torch.float16
        return "cuda", torch.float32
    return "cpu", torch.float32


def build_upstream_modules(local_model_dir: str, device: str, dtype: torch.dtype):
    vae = AutoencoderKLQwenImage.from_pretrained(local_model_dir, subfolder="vae")
    tokenizer = AutoTokenizer.from_pretrained(local_model_dir, subfolder="tokenizer")
    text_encoder = AutoModel.from_pretrained(local_model_dir, subfolder="text_encoder")

    vae = vae.to(device=device, dtype=dtype)
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    return vae, tokenizer, text_encoder


def main() -> None:
    args = parse_args()
    device, dtype = choose_device_and_dtype(args.mixed_precision)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    local_model_dir = snapshot_download(
        repo_id=args.model_name_or_path,
        allow_patterns=[
            "vae/*",
            "tokenizer/*",
            "text_encoder/*",
        ],
    )

    dataset = ImageCaptionDataset(
        root=args.train_data_dir,
        image_size=args.image_size,
        image_folder=args.image_folder,
        metadata_file=args.metadata_file,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=image_caption_collate,
    )

    vae, tokenizer, text_encoder = build_upstream_modules(
        local_model_dir=local_model_dir,
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

    sample_idx = 0
    for raw_batch in loader:
        processed = preprocessor.preprocess_batch(raw_batch)

        latents = processed["latents"].detach().cpu()
        prompt_embeds = processed["prompt_embeds"].detach().cpu()

        mask = processed["encoder_hidden_states_mask"]
        if mask is not None:
            mask = mask.detach().cpu()

        img_shapes = processed["img_shapes"]
        if isinstance(img_shapes, torch.Tensor):
            img_shapes = img_shapes.detach().cpu()

        file_names = processed["file_names"]
        prompts = processed["prompts"]

        batch_size = latents.shape[0]
        for i in range(batch_size):
            record = {
                "latents": latents[i],
                "prompt_embeds": prompt_embeds[i],
                "encoder_hidden_states_mask": None if mask is None else mask[i],
                "img_shapes": img_shapes[i] if isinstance(img_shapes, torch.Tensor) else img_shapes[i],
                "file_name": file_names[i],
                "prompt": prompts[i],
            }
            torch.save(record, output_dir / f"{sample_idx:06d}.pt")
            sample_idx += 1

    print(f"Saved {sample_idx} cached feature files to {output_dir}")


if __name__ == "__main__":
    main()