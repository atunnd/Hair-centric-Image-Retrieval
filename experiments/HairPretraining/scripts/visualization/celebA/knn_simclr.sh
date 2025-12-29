#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python knn_classification.py \
  --save_path visualization_output_dir_celebA \
  --size 224 \
  --train_annotation /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/data/data_train_combination3.csv \
  --test_annotation /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/data/data_test_combination3.csv \
  --img_dir /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/hair_regions \
  --mode simclr \
  --model resnet50 \
  --checkpoint_path /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/output_dir/simclr_resnet50/model_ckpt_latest.pth \
  --seed 42 \
  --num_workers 8 \
  --device cuda:3 \
  --batch_size 256 \
  --eval_type visualization

    