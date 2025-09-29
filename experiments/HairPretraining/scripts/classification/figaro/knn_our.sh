#!/bin/bash
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python knn_classification.py \
  --save_path classification_output_dir_Figaro \
  --size 224 \
  --train_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairPretraining/data/figaro_training.csv \
  --test_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairPretraining/data/figaro_testing.csv \
  --img_dir /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/Figaro/Figaro-1k/Total_hair \
  --mode our \
  --model resnet50 \
  --checkpoint_path "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairPretraining/output_dir/simclr_resnet50_neg_sample_supervised_mse_static_alpha/Copy of Copy of model_ckpt_299.pth" \
  --seed 42 \
  --num_workers 8 \
  --device cuda \
  --batch_size 256 \
  --our_method True \
  --eval_type linear_prob

    