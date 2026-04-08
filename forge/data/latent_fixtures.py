from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def _read_sd3_dims(model_name_or_path: str) -> dict[str, int]:
    root = Path(model_name_or_path)
    transformer_config = json.loads((root / "transformer" / "config.json").read_text(encoding="utf-8"))
    return {
        "in_channels": int(transformer_config["in_channels"]),
        "sample_size": int(transformer_config["sample_size"]),
        "joint_attention_dim": int(transformer_config["joint_attention_dim"]),
        "pooled_projection_dim": int(transformer_config["pooled_projection_dim"]),
    }


def build_sd3_dit_sample(model_name_or_path: str, index: int, seed: int = 42) -> dict[str, Any]:
    dims = _read_sd3_dims(model_name_or_path)
    base_seed = seed + index * 17

    latent_gen = torch.Generator(device="cpu").manual_seed(base_seed)
    prompt_gen = torch.Generator(device="cpu").manual_seed(base_seed + 1)
    pooled_gen = torch.Generator(device="cpu").manual_seed(base_seed + 2)
    noise_gen = torch.Generator(device="cpu").manual_seed(base_seed + 3)
    timestep_gen = torch.Generator(device="cpu").manual_seed(base_seed + 4)
    image_gen = torch.Generator(device="cpu").manual_seed(base_seed + 5)

    seq_len = 333
    latent = torch.randn(
        (dims["in_channels"], dims["sample_size"], dims["sample_size"]),
        generator=latent_gen,
        dtype=torch.float32,
    )
    prompt_embed = torch.randn((seq_len, dims["joint_attention_dim"]), generator=prompt_gen, dtype=torch.float32)
    pooled_prompt_embed = torch.randn((dims["pooled_projection_dim"],), generator=pooled_gen, dtype=torch.float32)
    noise = torch.randn(latent.shape, generator=noise_gen, dtype=torch.float32)
    timesteps = torch.randint(0, 1000, (1,), generator=timestep_gen, dtype=torch.long)
    image_embed = torch.randn((dims["pooled_projection_dim"],), generator=image_gen, dtype=torch.float32)
    attention_mask = torch.ones((seq_len,), dtype=torch.long)
    position_embedding = torch.zeros((1,), dtype=torch.float32)

    return {
        "sample_id": f"sample_{index:04d}",
        "latent": latent,
        "prompt_embed": prompt_embed,
        "pooled_prompt_embed": pooled_prompt_embed,
        "timesteps": timesteps,
        "noise": noise,
        "attention_mask": attention_mask,
        "position_embedding": position_embedding,
        "image_embed": image_embed,
    }


class LatentFixtureDataset(Dataset):
    def __init__(self, fixture_dir: str) -> None:
        self.fixture_dir = Path(fixture_dir)
        manifest_path = self.fixture_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Fixture manifest not found: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self.samples = [self.fixture_dir / item["file"] for item in manifest["samples"]]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return torch.load(self.samples[index], map_location="cpu")


def collate_latent_fixtures(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    tensor_keys = (
        "latent",
        "prompt_embed",
        "pooled_prompt_embed",
        "timesteps",
        "noise",
        "attention_mask",
        "position_embedding",
        "image_embed",
    )
    for key in tensor_keys:
        values = [item[key] for item in items if key in item]
        if values:
            batch[key] = torch.stack(values, dim=0)
    batch["sample_id"] = [item.get("sample_id", f"sample_{index:04d}") for index, item in enumerate(items)]
    return batch
