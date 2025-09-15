#!/bin/bash
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python mainpretrain.py \
  --epochs 300 \
  --batch_size 128 \
  --device cuda:0 \
  --device_id 2 \
  --save_path output_dir \
  --size 224 \
  --train_annotation data/data_train.csv\
  --test_annotation data/data_test.csv \
  --img_dir /data2/dragonzakura/QuocAnh/hair_regions/train/dummy_class \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --beta1 0.9 \
  --beta2 0.999 \
  --temp 0.7 \
  --mode simclr \
  --model resnet50 \
  --seed 42 \
  --num_workers 16 \
  --neg_sample True \
  --warm_up_epochs 20 \
  --neg_loss simclr \
  --sampling_frequency 50 \
  --fusion_type transformer


    