#!/bin/bash
export PYTORCH_ALLOC_CONF=expandable_segments:True

python mainpretrain.py \
  --epochs 300 \
  --batch_size 256 \
  --device cuda:4 \
  --device_id 2 \
  --save_path output_dir \
  --size 224 \
  --train_annotation data/data_train.csv\
  --test_annotation data/data_test.csv \
  --img_dir /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/hair_regions \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --beta1 0.9 \
  --beta2 0.999 \
  --temp 0.7 \
  --mode SHAM \
  --model vit_b_16 \
  --seed 42 \
  --num_workers 8 \
  --SHAM_mode reconstruction \
  --warm_up_epochs 20 \
  --sampling_frequency 30


    