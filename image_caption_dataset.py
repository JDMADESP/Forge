from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ImageCaptionDataset(Dataset):
    """
    class representing the image dataset

    Should have directory structure of :
        data/
            metadata.jsonl
            images/
                image1.png
                image2.png
    metadata.jnl should contain the following likes:
        {"file_name" : "image1.png", "text" : "car driving on the highway"}
    """
    def __init__(self, root : str, image_size : int = 1024, image_folder : str="images", metadata_file : str = "metadata.jsonl"):

        self.data_root = Path(root)
        if not self.data_root.exists():
            raise ValueError(f"Missing data folder: {root}")

        self.image_root = self.data_root / image_folder
        if not self.image_root.exists():
            raise ValueError(f"Missing image folder: {self.image_root}")

        self.metadata_path = self.data_root / metadata_file
        if not self.metadata_path.exists():
            raise ValueError(f"Missing metadata file: {self.metadata_path}")

        self.image_and_caption = []
        with self.metadata_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                info = json.loads(line)

                if "file_name" not in info or "text" not in info:
                    raise ValueError(f"The text, {line}, does not contain file_name or text")
                
                self.image_and_caption.append({
                    "file_name" : info["file_name"],
                    "text" : info["text"]
                    }
                )
        # need to normalize images in diffusion pipeline, need to check how to do this
        self.image_transform = transforms.Compose([
                                    transforms.Resize(
                                            image_size,
                                            interpolation=transforms.InterpolationMode.BILINEAR,
                                        ), # resizes image to image_size
                                    transforms.CenterCrop(image_size), # crops image at the center
                                    transforms.ToTensor(), # converts image to tensor
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), # normalizes image
                                ])

    def __len__(self):
        return len(self.image_and_caption)

    def __getitem__(self, index):
        info = self.image_and_caption[index]
        image_path = self.image_root / info["file_name"]

        if not image_path.exists():
            raise ValueError(f"Image file {image_path} does not exist")
        
        image = Image.open(image_path).convert("RGB")
        pixels = self.image_transform(image)

        return {
            "pixel_values" : pixels,
            "prompt" : info["text"],
            "file_name" : info["file_name"]
        }

def image_caption_collate(items):
    pixel_values = []
    prompts = []
    file_names = []

    for item in items:
        pixel_values.append(item["pixel_values"])
        prompts.append(item["prompt"])
        file_names.append(item["file_name"])
    pixel_values = torch.stack(pixel_values)
    return {
        "pixel_values" : pixel_values,
        "prompts" : prompts,
        "file_names" : file_names
    }