from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
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
