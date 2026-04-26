'''
Read the .pt files created by precompute_qwen_features.py and reutrn in batch format
training script will use
'''
from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset


class CachedQwenDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        if not self.root.exists():
            raise ValueError(f"Missing cached feature directory: {self.root}")

        self.files = sorted(self.root.glob("*.pt"))
        if not self.files:
            raise ValueError(f"No .pt files found in {self.root}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int):
        return torch.load(self.files[index], map_location="cpu")


def cached_qwen_collate(items):
    latents = torch.stack([item["latents"] for item in items], dim=0)
    prompt_embeds = torch.stack([item["prompt_embeds"] for item in items], dim=0)

    masks = [item["encoder_hidden_states_mask"] for item in items]
    encoder_hidden_states_mask = None
    if masks[0] is not None:
        encoder_hidden_states_mask = torch.stack(masks, dim=0)

    img_shapes_items = [item["img_shapes"] for item in items]
    if isinstance(img_shapes_items[0], torch.Tensor):
        img_shapes = torch.stack(img_shapes_items, dim=0)
    else:
        img_shapes = img_shapes_items

    return {
        "latents": latents,
        "prompt_embeds": prompt_embeds,
        "encoder_hidden_states_mask": encoder_hidden_states_mask,
        "img_shapes": img_shapes,
        "file_names": [item["file_name"] for item in items],
        "prompts": [item["prompt"] for item in items],
    }