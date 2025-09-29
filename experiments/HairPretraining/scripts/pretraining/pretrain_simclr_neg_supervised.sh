#!/bin/bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python mainpretrain.py \
  --epochs 300 \
  --batch_size 256 \
  --device cuda:0 \
  --device_id 2 \
  --save_path output_dir \
  --size 224 \
  --train_annotation data/data_train.csv\
  --test_annotation data/data_test.csv \
  --img_dir /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairClassification/data/hair_regions/train/dummy_class \
  --lr 0.001 \
  --weight_decay 0.0001 \
  --beta1 0.9 \
  --beta2 0.999 \
  --temp 0.7 \
  --mode simclr \
  --model vit_b_16 \
  --seed 42 \
  --num_workers 16 \
  --neg_sample True \
  --warm_up_epochs 20 \
  --neg_loss simclr \
  --sampling_frequency 50 \


    