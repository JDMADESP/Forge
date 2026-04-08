from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrainingEngineArgs:
    model_name_or_path: str
    output_dir: str = "outputs/m0_sd3_fsdp"
    train_fixture_dir: str | None = None
    train_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-5
    weight_decay: float = 1e-2
    max_train_steps: int = 1000
    num_train_epochs: int = 1
    checkpoint_every_n_steps: int = 100
    validation_every_n_steps: int = 100
    # used for grad clipping
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"
    seed: int = 42
    # which gpu to use, -1 for cpu
    local_rank: int = -1
    dataloader_num_workers: int = 0
    resume_from_checkpoint: str | None = None
    # if to record sample images during training, requires validation prompts to be set
    log_samples: bool = True
    validation_prompts: list[str] = field(
        default_factory=lambda: ["a clean OCR-style poster that says FORGE"])

    def output_path(self) -> Path:
        return Path(self.output_dir)
