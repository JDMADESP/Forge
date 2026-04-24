'''
Running to check if dataset can be processed properly
'''
from torch.utils.data import DataLoader

from image_caption_dataset import ImageCaptionDataset, image_caption_collate

def main():
    dataset = ImageCaptionDataset(
        root="toy_data",
        image_size=256,
        image_folder="images",
        metadata_file="metadata.jsonl",
    )

    print("dataset length:", len(dataset))

    sample = dataset[0]
    print("single sample keys:", sample.keys())
    print("single image shape:", sample["pixel_values"].shape)
    print("single prompt:", sample["prompt"])
    print("single file name:", sample["file_name"])

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
        collate_fn=image_caption_collate,
    )

    batch = next(iter(loader))
    print("batch keys:", batch.keys())
    print("batch pixel_values shape:", batch["pixel_values"].shape)
    print("batch prompts:", batch["prompts"])
    print("batch file_names:", batch["file_names"])

if __name__ == "__main__":
    main()