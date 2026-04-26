from __future__ import annotations

import argparse
import os
import random

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from huggingface_hub import snapshot_download

from forge.core import create_core
from forge.trainer import Trainer
from forge.training_args import TrainingEngineArgs

from cached_qwen_dataset import CachedQwenDataset, cached_qwen_collate


def build_optimizer(model: torch.nn.Module, args: TrainingEngineArgs) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Qwen from cached features")
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--cached_data_dir", required=True)
    parser.add_argument("--output_dir", default="outputs/qwen_cached")
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
    parser.add_argument("--dataloader_num_workers", type=int, default=0)

    parser.add_argument("--parallel_backend", default="torch")
    parser.add_argument("--dp_mode", default="fsdp1")
    parser.add_argument("--sequence_parallel_mode", default="none")
    parser.add_argument("--sequence_parallel_algorithm", default="ulysses")
    parser.add_argument("--sequence_parallel_degree", type=int, default=1)
    parser.add_argument("--attention_backend", default="native")

    return parser.parse_args()


def args_to_training_engine(cli_args: argparse.Namespace) -> TrainingEngineArgs:
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
        dataloader_num_workers=cli_args.dataloader_num_workers,
        resume_from_checkpoint=cli_args.resume_from_checkpoint,
        parallel_backend=cli_args.parallel_backend,
        dp_mode=cli_args.dp_mode,
        sequence_parallel_mode=cli_args.sequence_parallel_mode,
        sequence_parallel_algorithm=cli_args.sequence_parallel_algorithm,
        sequence_parallel_degree=cli_args.sequence_parallel_degree,
        attention_backend=cli_args.attention_backend,
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


def main() -> None:
    cli_args = parse_args()
    init_dist()

    try:
        args = args_to_training_engine(cli_args)
        set_seed(args.seed)

        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        world_size = int(os.environ.get("WORLD_SIZE", "1"))

        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)

        local_model_dir = snapshot_download(
            repo_id=cli_args.model_name_or_path,
            allow_patterns=[
                "transformer/*",
                "scheduler/*",
            ],
        )
        args.model_name_or_path = local_model_dir

        if local_rank == 0:
            print(
                f"[launch] world_size={world_size} "
                f"dp_mode={args.dp_mode} "
                f"sp_mode={args.sequence_parallel_mode} "
                f"sp_degree={args.sequence_parallel_degree}",
                flush=True,
            )

        dataset = CachedQwenDataset(cli_args.cached_data_dir)

        sampler = None
        if dist.is_available() and dist.is_initialized():
            sampler = DistributedSampler(dataset, shuffle=True)

        train_loader = DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=False if sampler is not None else True,
            sampler=sampler,
            num_workers=args.dataloader_num_workers,
            pin_memory=torch.cuda.is_available(),
            collate_fn=cached_qwen_collate,
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