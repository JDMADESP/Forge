Current Workflow
Download and build coco_dataset:
python download_and_build_coco_subset.py \
  --workspace_dir /workspace/coco_work \
  --subset_output_name coco_subset_1000 \
  --images_zip_url "http://images.cocodataset.org/zips/train2017.zip" \
  --captions_zip_url "http://images.cocodataset.org/annotations/annotations_trainval2017.zip" \
  --images_folder_name train2017 \
  --captions_json_relpath annotations/captions_train2017.json \
  --max_images 1000 \
  --captions_per_image 1 \
  --selection_mode random \
  --link_mode symlink

 Precompute the features:
  python precompute_qwen_features.py \
  --model_name_or_path Qwen/Qwen-Image \
  --train_data_dir /workspace/coco_work/coco_subset_1000 \
  --output_dir /workspace/coco_work/cached_coco_subset_1000 \
  --image_size 256 \
  --batch_size 1 \
  --mixed_precision fp32

  Train from precomputer features:
  python train_qwen_cached.py \
  --model_name_or_path Qwen/Qwen-Image \
  --cached_data_dir /workspace/coco_work/cached_coco_subset_1000 \
  --train_batch_size 1 \
  --max_train_steps 1 \
  --num_train_epochs 1 \
  --mixed_precision bf16 \
  --output_dir outputs/qwen_coco_subset