#!/bin/bash

python mainpretrain.py \
  --epochs 20 \
  --batch_size 32 \
  --device cpu \
  --save_path output_dir \
  --size 224 \
  --train_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairClassification/data_analysis/data_train.csv\
  --test_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairClassification/data_analysis/data_test.csv \
  --img_dir /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairClassification/data/hair_regions \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --beta1 0.9 \
  --beta2 0.999 \
  --temp 0.7 \
  --mode simclr_supcon \
  --model resnet18 \
  --seed 42 \
  --num_workers 4 \
  --test True \
  --test_model_path /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairPretraining/output_dir/simclr_supcon/model_ckpt_8.pth