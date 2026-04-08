from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Any, Iterable

import torch

from forge.adapters.base import ModelAdapter
from forge.parallel.base import ParallelStrategy
from forge.training_args import TrainingEngineArgs


@dataclass
class TrainerState:
    epoch: int = 0
    global_step: int = 0
    best_metric: float | None = None


class Trainer:
    def __init__(
        self,
        args: TrainingEngineArgs,
        model_adapter: ModelAdapter,
        strategy: ParallelStrategy,
        train_loader: Iterable[dict[str, Any]],
        optimizer_factory: Any,
        scheduler_factory: Any | None = None,
    ) -> None:
        self.args = args
        self.model_adapter = model_adapter
        self.strategy = strategy
        self.train_loader = train_loader
        self.optimizer_factory = optimizer_factory
        self.scheduler_factory = scheduler_factory
        self.state = TrainerState()

        self.modules = self.model_adapter.build_modules(args)
        self.model = self.modules["model"]
        self.model = self.strategy.prepare_model(self.model)
        self.model_adapter.set_train_model(self.model)
        self.optimizer = self.strategy.prepare_optimizer(
            self.model, self.optimizer_factory(self.model)
        )
        self.scheduler = (
            self.scheduler_factory(self.optimizer) if self.scheduler_factory else None
        )

        if args.resume_from_checkpoint:
            loaded = self.strategy.load(
                args.resume_from_checkpoint, self.model, self.optimizer
            )
            self.state.epoch = int(loaded.get("epoch", 0))
            self.state.global_step = int(loaded.get("global_step", 0))
            self._load_rng_state(args.resume_from_checkpoint)

    def train(self) -> None:
        self.args.output_path().mkdir(parents=True, exist_ok=True)
        self.model.train()
        accumulation = max(1, self.args.gradient_accumulation_steps)
        micro_step = 0
        resume_micro_step = self.state.global_step * accumulation

        for epoch in range(self.state.epoch, self.args.num_train_epochs):
            self.state.epoch = epoch
            sampler = getattr(self.train_loader, "sampler", None)
            if sampler is not None and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            for raw_batch in self.train_loader:
                if resume_micro_step > 0:
                    resume_micro_step -= 1
                    continue
                batch = self.model_adapter.prepare_batch(raw_batch)
                out = self.model_adapter.forward_loss(batch)
                loss = out["loss"] / accumulation
                self.strategy.backward(loss)
                micro_step += 1

                if micro_step % accumulation == 0:
                    self.strategy.clip_grad_norm_(self.model, self.args.max_grad_norm)
                    self.strategy.step(self.optimizer, self.scheduler)
                    self.state.global_step += 1
                    loss_value = out.get("metrics", {}).get("loss")
                    if loss_value is None:
                        loss_value = float(loss.detach().item() * accumulation)
                    if self.strategy.is_main_process():
                        print(f"[train] step={self.state.global_step} loss={loss_value:.6f}", flush=True)

                    if self._should_validate():
                        self.validate()
                    if self._should_checkpoint():
                        self.save_checkpoint()
                    if self.state.global_step >= self.args.max_train_steps:
                        return

    def validate(self) -> dict[str, Any]:
        self.model.eval()
        try:
            return self.model_adapter.validation_generate(
                prompts=self.args.validation_prompts,
                output_dir=str(self.args.output_path() / "validation"),
                seed=self.args.seed,
            )
        finally:
            self.model.train()

    def save_checkpoint(self) -> None:
        ckpt_dir = self.args.output_path() / f"checkpoint-{self.state.global_step}"
        payload = {
            "epoch": self.state.epoch,
            "global_step": self.state.global_step,
        }
        self.strategy.save(str(ckpt_dir), payload)
        if self.strategy.is_main_process():
            self._save_rng_state(ckpt_dir)

    def _should_checkpoint(self) -> bool:
        every = self.args.checkpoint_every_n_steps
        return every > 0 and self.state.global_step % every == 0

    def _should_validate(self) -> bool:
        every = self.args.validation_every_n_steps
        return every > 0 and self.state.global_step % every == 0

    def _save_rng_state(self, ckpt_dir: Path) -> None:
        rng_state = {
            "python_random_state": random.getstate(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state_all": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        }
        torch.save(rng_state, ckpt_dir / "rng_state.pt")

    def _load_rng_state(self, ckpt_dir: str) -> None:
        rng_path = Path(ckpt_dir) / "rng_state.pt"
        if not rng_path.exists():
            return
        rng_state = torch.load(rng_path, map_location="cpu")
        python_random_state = rng_state.get("python_random_state")
        if python_random_state is not None:
            random.setstate(python_random_state)

        torch_rng_state = rng_state.get("torch_rng_state")
        if torch_rng_state is not None:
            torch.set_rng_state(torch_rng_state)

        cuda_rng_state_all = rng_state.get("cuda_rng_state_all")
        if cuda_rng_state_all is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(cuda_rng_state_all)
