from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download COCO remotely, extract it, and build a subset in metadata.jsonl + images/ format."
    )

    parser.add_argument(
        "--workspace_dir",
        required=True,
        help="Base directory where downloads, extracted data, and subset output will live.",
    )
    parser.add_argument(
        "--subset_output_name",
        default="coco_subset",
        help="Name of the final subset folder inside workspace_dir.",
    )

    parser.add_argument(
        "--images_zip_url",
        required=True,
        help="Remote URL for COCO images zip, e.g. train2017.zip.",
    )
    parser.add_argument(
        "--captions_zip_url",
        required=True,
        help="Remote URL for COCO captions annotations zip, e.g. annotations_trainval2017.zip.",
    )

    parser.add_argument(
        "--images_folder_name",
        default="train2017",
        help="Expected extracted images folder name inside the workspace.",
    )
    parser.add_argument(
        "--captions_json_relpath",
        default="annotations/captions_train2017.json",
        help="Relative path to captions json after extracting the annotations zip.",
    )

    parser.add_argument(
        "--max_images",
        type=int,
        default=1000,
        help="Maximum number of unique images to include in the subset.",
    )
    parser.add_argument(
        "--captions_per_image",
        type=int,
        default=1,
        help="How many captions to keep per image.",
    )
    parser.add_argument(
        "--selection_mode",
        choices=["random", "first"],
        default="random",
        help="How to choose images from COCO.",
    )
    parser.add_argument(
        "--link_mode",
        choices=["copy", "symlink", "hardlink"],
        default="symlink",
        help="How to place images into the final subset images/ folder.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="HTTP timeout in seconds.",
    )
    parser.add_argument(
        "--chunk_size_mb",
        type=int,
        default=8,
        help="Download chunk size in MB.",
    )

    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def stream_download(url: str, destination: Path, timeout: int, chunk_size_mb: int) -> None:
    if destination.exists() and destination.stat().st_size > 0:
        print(f"Skipping download, file already exists: {destination}")
        return

    ensure_dir(destination.parent)
    chunk_size = chunk_size_mb * 1024 * 1024

    print(f"Downloading: {url}")
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        total_bytes = int(response.headers.get("content-length", 0))
        downloaded = 0

        with destination.open("wb") as f:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                f.write(chunk)
                downloaded += len(chunk)

                if total_bytes > 0:
                    pct = 100.0 * downloaded / total_bytes
                    print(
                        f"\r  downloaded {downloaded / 1e9:.2f} GB / {total_bytes / 1e9:.2f} GB "
                        f"({pct:.1f}%)",
                        end="",
                        flush=True,
                    )

    if total_bytes > 0:
        print()
    print(f"Saved: {destination}")


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    ensure_dir(extract_to)
    print(f"Extracting: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"Extracted into: {extract_to}")


def load_coco_annotations(annotations_json: Path) -> tuple[dict[int, dict], dict[int, list[str]]]:
    with annotations_json.open("r", encoding="utf-8") as f:
        coco = json.load(f)

    images_by_id: dict[int, dict] = {}
    for image_info in coco["images"]:
        images_by_id[int(image_info["id"])] = image_info

    captions_by_image_id: dict[int, list[str]] = defaultdict(list)
    for ann in coco["annotations"]:
        image_id = int(ann["image_id"])
        caption = ann["caption"].strip()
        if caption:
            captions_by_image_id[image_id].append(caption)

    return images_by_id, captions_by_image_id


def place_image(src: Path, dst: Path, link_mode: str) -> None:
    if dst.exists() or dst.is_symlink():
        dst.unlink()

    if link_mode == "copy":
        shutil.copy2(src, dst)
    elif link_mode == "symlink":
        os.symlink(src.resolve(), dst)
    elif link_mode == "hardlink":
        os.link(src, dst)
    else:
        raise ValueError(f"Unsupported link_mode: {link_mode}")


def build_subset(
    images_dir: Path,
    annotations_json: Path,
    subset_dir: Path,
    max_images: int,
    captions_per_image: int,
    selection_mode: str,
    link_mode: str,
    seed: int,
) -> None:
    random.seed(seed)

    subset_images_dir = subset_dir / "images"
    metadata_path = subset_dir / "metadata.jsonl"

    ensure_dir(subset_images_dir)

    images_by_id, captions_by_image_id = load_coco_annotations(annotations_json)

    candidate_image_ids = [
        image_id
        for image_id in images_by_id
        if image_id in captions_by_image_id and len(captions_by_image_id[image_id]) > 0
    ]

    if selection_mode == "random":
        random.shuffle(candidate_image_ids)
    else:
        candidate_image_ids = sorted(candidate_image_ids)

    selected_image_ids = candidate_image_ids[:max_images]

    num_records = 0
    num_images_kept = 0

    with metadata_path.open("w", encoding="utf-8") as f:
        for image_id in selected_image_ids:
            image_info = images_by_id[image_id]
            file_name = image_info["file_name"]
            src_image_path = images_dir / file_name

            if not src_image_path.exists():
                print(f"Skipping missing image: {src_image_path}")
                continue

            dst_image_path = subset_images_dir / file_name
            place_image(src_image_path, dst_image_path, link_mode)
            num_images_kept += 1

            captions = list(captions_by_image_id[image_id])
            if selection_mode == "random":
                random.shuffle(captions)

            kept_captions = captions[:captions_per_image]
            for caption in kept_captions:
                record = {
                    "file_name": file_name,
                    "text": caption,
                    "coco_image_id": image_id,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                num_records += 1

    print(f"Built subset at: {subset_dir}")
    print(f"Images kept: {num_images_kept}")
    print(f"Metadata records: {num_records}")
    print(f"Images dir: {subset_images_dir}")
    print(f"Metadata file: {metadata_path}")


def main() -> None:
    args = parse_args()

    workspace_dir = Path(args.workspace_dir)
    downloads_dir = workspace_dir / "downloads"
    extracted_dir = workspace_dir / "extracted"
    subset_dir = workspace_dir / args.subset_output_name

    ensure_dir(downloads_dir)
    ensure_dir(extracted_dir)

    images_zip_path = downloads_dir / Path(args.images_zip_url).name
    captions_zip_path = downloads_dir / Path(args.captions_zip_url).name

    stream_download(
        url=args.images_zip_url,
        destination=images_zip_path,
        timeout=args.timeout,
        chunk_size_mb=args.chunk_size_mb,
    )
    stream_download(
        url=args.captions_zip_url,
        destination=captions_zip_path,
        timeout=args.timeout,
        chunk_size_mb=args.chunk_size_mb,
    )

    images_dir = extracted_dir / args.images_folder_name
    annotations_json = extracted_dir / args.captions_json_relpath

    if not images_dir.exists():
        extract_zip(images_zip_path, extracted_dir)
    else:
        print(f"Skipping image extraction, folder already exists: {images_dir}")

    if not annotations_json.exists():
        extract_zip(captions_zip_path, extracted_dir)
    else:
        print(f"Skipping annotation extraction, file already exists: {annotations_json}")

    if not images_dir.exists():
        raise ValueError(f"Expected extracted images folder not found: {images_dir}")
    if not annotations_json.exists():
        raise ValueError(f"Expected captions json not found: {annotations_json}")

    build_subset(
        images_dir=images_dir,
        annotations_json=annotations_json,
        subset_dir=subset_dir,
        max_images=args.max_images,
        captions_per_image=args.captions_per_image,
        selection_mode=args.selection_mode,
        link_mode=args.link_mode,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()