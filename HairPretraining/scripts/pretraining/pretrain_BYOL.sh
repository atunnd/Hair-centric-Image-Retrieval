#!/bin/bash

python mainpretrain.py \
  --epochs 300 \
  --batch_size 256 \
  --device cuda \
  --device_id 2 \
  --save_path output_dir \
  --size 224 \
  --train_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/HairPretraining/data/data_train.csv \
  --test_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/HairPretraining/data/data_test.csv \
  --img_dir /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairClassification/data/hair_regions/train/dummy_class \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --beta1 0.9 \
  --beta2 0.999 \
  --mode BYOL \
  --model resnet50 \
  --seed 42 \
  --num_workers 16 \



    