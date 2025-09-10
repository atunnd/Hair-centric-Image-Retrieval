#!/bin/bash
python mainpretrain.py \
  --epochs 300 \
  --batch_size 256 \
  --device cuda \
  --device_id 2 \
  --save_path output_dir \
  --size 224 \
  --train_annotation data/data_train.csv\
  --test_annotation data/data_test.csv \
  --img_dir /data2/dragonzakura/QuocAnh/hair_regions \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --beta1 0.9 \
  --beta2 0.999 \
  --temp 0.7 \
  --mode dinov2 \
  --model vit_b_16 \
  --seed 42 \
  --num_workers 8 \


    