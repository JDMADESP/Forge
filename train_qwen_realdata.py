from __future__ import annotations

import argparse
import os
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from functools import partial

from diffusers import AutoencoderKL
from transformers import AutoTokenizer, AutoModel

from forge.core import create_core
from forge.data import LatentFixtureDataset, collate_latent_fixtures
from forge.trainer import Trainer
from forge.training_args import TrainingEngineArgs

from image_caption_dataset import ImageCaptionDataset, image_caption_collate
from preprocess_for_qwen import QwenImagePreProcessor


def build_optimizer(model: torch.nn.Module, args: TrainingEngineArgs) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M2 Qwen Training")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--output_dir", default="outputs/qwen_realdata")
    parser.add_argument("--train_fixture_dir", required=True)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--max_train_steps", type=int, default=10)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--checkpoint_every_n_steps", type=int, default=5)
    parser.add_argument("--validation_every_n_steps", type=int, default=5)
    parser.add_argument("--mixed_precision", default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from_checkpoint", default=None)

    # data preprocessing args
    parser.add_argument("--train_data_dir", required=True)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--image_folder", default="images")
    parser.add_argument("--metadata_file", default="metadata.jsonl")
    parser.add_argument("--vae_name_or_path", default=None)
    parser.add_argument("--tokenizer_name_or_path", default=None)
    parser.add_argument("--text_encoder_name_or_path", default=None)

    return parser.parse_args()

def args_to_training_engine(cli_args : argparse.Namespace) -> TrainingEngineArgs:
    return TrainingEngineArgs(
        model_name_or_path=cli_args.model_name_or_path,
        model_family="qwen_image",
        output_dir=cli_args.output_dir,
        train_batch_size=cli_args.train_batch_size,
        gradient_accumulation_steps=cli_args.gradient_accumulation_steps,
        learning_rate=cli_args.learning_rate,
        weight_decay=cli_args.weight_decay,
        max_train_steps=cli_args.max_train_steps,
        num_train_epochs=cli_args.num_train_epochs,
        checkpoint_every_n_steps=cli_args.checkpoint_every_n_steps,
        validation_every_n_steps=cli_args.validation_every_n_steps,
        mixed_precision=cli_args.mixed_precision,
        seed=cli_args.seed,
        resume_from_checkpoint=cli_args.resume_from_checkpoint,
    )

def init_dist() -> None:
    if dist.is_initialized():
        return
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size == 1:
        return
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if backend == "gloo":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo")
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29500")
    dist.init_process_group(backend=backend)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def build_upstream_modules(model_name_or_path : str, 
                           vae_name_or_path : str | None, 
                           tokenizer_name_or_path : str | None, 
                           text_encoder_name_or_path : str | None, 
                           device : str, 
                           dtype : torch.dtype
    ):
    vae_source = vae_name_or_path or model_name_or_path
    tokenizer_source = tokenizer_name_or_path or model_name_or_path
    text_encoder_source = text_encoder_name_or_path or model_name_or_path

    vae = AutoencoderKL.from_pretrained(vae_source)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    text_encoder = AutoModel.from_pretrained(text_encoder_source)

    vae = vae.to(device=device, dtype=dtype)
    text_encoder = text_encoder.to(device=device, dtype=dtype)

    return vae, tokenizer, text_encoder

def preprocess_collate_fn(items, preprocessor: QwenImagePreProcessor):
    raw_batch = image_caption_collate(items)
    processed_batch = preprocessor.preprocess_batch(raw_batch)
    return processed_batch

def main() -> None:
    args = parse_args()
    init_dist()
    try:
        args = args_to_training_engine(args)
        set_seed(args.seed)
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if cli_args.mixed_precision == "bf16" and torch.cuda.is_available() else torch.float32

        dataset = ImageCaptionDataset(
            root=cli_args.train_data_dir,
            image_size=cli_args.image_size,
            image_folder=cli_args.image_folder,
            metadata_file=cli_args.metadata_file,
        )

        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=False)

        vae, tokenizer, text_encoder = build_upstream_modules(
            model_name_or_path=cli_args.model_name_or_path,
            vae_name_or_path=cli_args.vae_name_or_path,
            tokenizer_name_or_path=cli_args.tokenizer_name_or_path,
            text_encoder_name_or_path=cli_args.text_encoder_name_or_path,
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

        collate_fn = partial(preprocess_collate_fn, preprocessor=preprocessor)

        train_loader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=False if sampler is not None else False,
            sampler=sampler,
            num_workers=args.dataloader_num_workers,
            collate_fn=collate_fn,
        )

        core = create_core(
            model_family=args.model_family,
            model_name_or_path=args.model_name_or_path,
            parallel_config=args.parallel_config(),
        )
        trainer = Trainer(
            args=args,
            core=core,
            train_loader=train_loader,
            optimizer_factory=lambda model: build_optimizer(model, args),
        )
        trainer.train()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
