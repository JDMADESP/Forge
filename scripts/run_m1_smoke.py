from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import torch
import transformers
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def bootstrap_env() -> None:
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
            def _is_in_bad_fork() -> bool:
                return False

            @staticmethod
            def manual_seed_all(seed: int) -> None:
                del seed

            @staticmethod
            def is_available() -> bool:
                return False

            @staticmethod
            def synchronize() -> None:
                return None

        torch.xpu = _XpuStub()  # type: ignore[attr-defined]

    if not hasattr(torch, "library"):
        class _LibraryStub:
            @staticmethod
            def register_fake(*args: object, **kwargs: object):
                del args, kwargs

                def decorator(fn: object) -> object:
                    return fn

                return decorator

        torch.library = _LibraryStub()  # type: ignore[attr-defined]
    elif not hasattr(torch.library, "register_fake"):
        def _register_fake(*args: object, **kwargs: object):
            del args, kwargs

            def decorator(fn: object) -> object:
                return fn

            return decorator

        torch.library.register_fake = _register_fake  # type: ignore[attr-defined]

    if not hasattr(transformers, "AutoImageProcessor"):
        class _AutoImageProcessorStub:
            @classmethod
            def from_pretrained(cls, *args: object, **kwargs: object) -> "_AutoImageProcessorStub":
                del args, kwargs
                return cls()

        transformers.AutoImageProcessor = _AutoImageProcessorStub  # type: ignore[attr-defined]

    if not hasattr(transformers, "PreTrainedModel"):
        try:
            from transformers.modeling_utils import PreTrainedModel as _PreTrainedModel
        except Exception:
            class _PreTrainedModel:
                pass
        else:
            transformers.PreTrainedModel = _PreTrainedModel  # type: ignore[attr-defined]


bootstrap_env()

from forge.core import create_core
from forge.data import LatentFixtureDataset, collate_latent_fixtures
from forge.trainer import Trainer
from forge.training_args import TrainingEngineArgs


def build_optimizer(model: torch.nn.Module, args: TrainingEngineArgs) -> torch.optim.Optimizer:
    params = [param for param in model.parameters() if param.requires_grad]
    return torch.optim.AdamW(
        params,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="M1 SD3 single-card smoke run")
    parser.add_argument(
        "--model_name_or_path",
        default="models/stable-diffusion-3-medium-diffusers",
    )
    parser.add_argument(
        "--train_fixture_dir",
        default="M0_legacy/test_M0/outputs/l0_fixtures",
    )
    parser.add_argument("--output_dir", default="outputs/m1_smoke")
    parser.add_argument("--max_train_steps", type=int, default=3)
    parser.add_argument("--train_batch_size", type=int, default=1)
    parser.add_argument("--mixed_precision", default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main() -> None:
    cli_args = parse_args()
    set_seed(cli_args.seed)
    if torch.cuda.is_available():
        torch.cuda.set_device(0)

    args = TrainingEngineArgs(
        model_name_or_path=cli_args.model_name_or_path,
        train_fixture_dir=cli_args.train_fixture_dir,
        output_dir=cli_args.output_dir,
        max_train_steps=cli_args.max_train_steps,
        train_batch_size=cli_args.train_batch_size,
        mixed_precision=cli_args.mixed_precision,
        seed=cli_args.seed,
        checkpoint_every_n_steps=0,
        validation_every_n_steps=0,
    )

    dataset = LatentFixtureDataset(args.train_fixture_dir)
    train_loader = DataLoader(
        dataset,
        batch_size=args.train_batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_latent_fixtures,
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


if __name__ == "__main__":
    main()
