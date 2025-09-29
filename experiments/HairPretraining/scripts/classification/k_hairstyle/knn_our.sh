#!/bin/bash
#export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python knn_classification.py \
  --save_path classification_output_dir_K-hairstyle \
  --size 224 \
  --train_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/k-hairstyle_classification/classification_training_korean_hairstyle_benchmark/training_classification_labels.csv \
  --test_annotation /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/k-hairstyle_classification/classification_testing_korean_hairstyle_benchmark/testing_classification_labels.csv \
  --img_dir /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/k-hairstyle_classification/total_hair_regions \
  --mode our \
  --model resnet50 \
  --checkpoint_path "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/HairPretraining/output_dir/simclr_resnet50_neg_sample_supervised_mse_static_alpha/Copy of Copy of model_ckpt_299.pth" \
  --seed 42 \
  --num_workers 8 \
  --device cuda \
  --batch_size 256 \
  --our_method True \
  --eval_type linear_prob

    