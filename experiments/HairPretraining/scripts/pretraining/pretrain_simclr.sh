#!/bin/bash
export PYTORCH_ALLOC_CONF=expandable_segments:True
ulimit -n 65535


python mainpretrain.py \
  --epochs 300 \
  --batch_size 256 \
  --device cuda \
  --device_id 2 \
  --save_path output_dir \
  --size 224 \
  --train_annotation /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/data/data_train.csv \
  --test_annotation /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/data/data_test.csv \
  --img_dir /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/hair_regions \
  --lr 0.0001 \
  --weight_decay 0.0001 \
  --beta1 0.9 \
  --beta2 0.999 \
  --temp 0.7 \
  --mode simclr \
  --model resnet50 \
  --seed 42 \
  --num_workers 8 \
  # --full_face_training \
  # --continue_training \
  # --checkpoint_folder /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/output_dir/simclr_resnet50_full_face_training




    