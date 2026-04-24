from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from diffusers import DiffusionPipeline

from forge.core import create_core
from forge.training_args import TrainingEngineArgs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate Qwen real-data fine-tuning")

    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--checkpoint_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/qwen_eval")

    parser.add_argument("--prompts", nargs="*", default=None)
    parser.add_argument("--prompts_file", default=None)

    parser.add_argument("--num_inference_steps", type=int, default=20)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--mixed_precision", default="fp32")
    parser.add_argument("--local_files_only", action="store_true")

    return parser.parse_args()

def resolve_prompts(args: argparse.Namespace) -> list[str]:
    if args.prompts:
        return args.prompts

    if args.prompts_file is not None:
        prompt_path = Path(args.prompts_file)
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}")

        prompts = []
        for line in prompt_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                prompts.append(line)

        if not prompts:
            raise ValueError(f"No prompts found in {prompt_path}")

        return prompts

    return [
        "a clean OCR-style poster that says FORGE",
        "a red car on the road",
        "a blue sky over mountains",
    ]

def choose_device_and_dtype(mixed_precision: str) -> tuple[str, torch.dtype]:
    if torch.cuda.is_available():
        if mixed_precision == "bf16":
            return "cuda", torch.bfloat16
        if mixed_precision == "fp16":
            return "cuda", torch.float16
        return "cuda", torch.float32

    return "cpu", torch.float32

def snapshot_local_model(model_name_or_path: str, local_files_only: bool = False) -> str:
    return snapshot_download(
        repo_id=model_name_or_path,
        allow_patterns=[
            "model_index.json",
            "scheduler/*",
            "vae/*",
            "tokenizer/*",
            "text_encoder/*",
            "transformer/*",
        ],
        local_files_only=local_files_only,
    )

def build_base_pipeline(
    local_model_dir: str,
    device: str,
    dtype: torch.dtype,
):
    pipe = DiffusionPipeline.from_pretrained(
        local_model_dir,
        torch_dtype=dtype,
        local_files_only=True,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=False)
    return pipe


def build_finetuned_transformer(
    local_model_dir: str,
    checkpoint_dir: str,
):
    args = TrainingEngineArgs(
        model_name_or_path=local_model_dir,
        model_family="qwen_image",
        output_dir="outputs/debug_eval",
        mixed_precision="fp32",
    )

    core = create_core(
        model_family=args.model_family,
        model_name_or_path=args.model_name_or_path,
        parallel_config=args.parallel_config(),
    )

    model = core.model_runtime.build_model()
    core.model_runtime.load_weights(model)

    parallel_plan = core.model_runtime.make_parallel_plan(args.parallel_config())
    core.parallel_runtime.setup(parallel_plan)
    model = core.parallel_runtime.parallelize_model(model, parallel_plan)

    optimizer = core.parallel_runtime.prepare_optimizer(
        model,
        torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=1e-5,
        ),
    )

    core.parallel_runtime.load(
        checkpoint_dir,
        model,
        optimizer,
    )

    model.eval()
    return model


def save_generation_batch(
    pipe,
    prompts: list[str],
    output_dir: Path,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    seed: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, prompt in enumerate(prompts):
        generator = torch.Generator(device=pipe.device).manual_seed(seed + idx)

        result = pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator,
        )

        image = result.images[0]
        image_path = output_dir / f"{idx:02d}.png"
        txt_path = output_dir / f"{idx:02d}.txt"

        image.save(image_path)
        txt_path.write_text(prompt, encoding="utf-8")


def main() -> None:
    args = parse_args()
    prompts = resolve_prompts(args)

    device, dtype = choose_device_and_dtype(args.mixed_precision)

    local_model_dir = snapshot_local_model(
        args.model_name_or_path,
        local_files_only=args.local_files_only,
    )

    output_root = Path(args.output_dir)
    base_dir = output_root / "base"
    finetuned_dir = output_root / "finetuned"

    base_pipe = build_base_pipeline(local_model_dir, device=device, dtype=dtype)
    save_generation_batch(
        pipe=base_pipe,
        prompts=prompts,
        output_dir=base_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )

    finetuned_transformer = build_finetuned_transformer(
        local_model_dir=local_model_dir,
        checkpoint_dir=args.checkpoint_dir,
    )

    finetuned_pipe = build_base_pipeline(local_model_dir, device=device, dtype=dtype)
    finetuned_pipe.transformer = finetuned_transformer

    save_generation_batch(
        pipe=finetuned_pipe,
        prompts=prompts,
        output_dir=finetuned_dir,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        height=args.height,
        width=args.width,
        seed=args.seed,
    )

    print(f"Saved base outputs to: {base_dir}")
    print(f"Saved finetuned outputs to: {finetuned_dir}")


if __name__ == "__main__":
    main()