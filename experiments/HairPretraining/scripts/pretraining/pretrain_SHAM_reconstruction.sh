#!/bin/bash
export PYTORCH_ALLOC_CONF=expandable_segments:True


python mainpretrain.py \
  --epochs 300 \
  --batch_size 256 \
  --device cuda \
  --device_id 3 \
  --save_path output_dir \
  --size 224 \
  --train_annotation /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/data/data_train.csv \
  --test_annotation /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/data/data_test.csv \
  --img_dir /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/hair_regions \
  --lr 0.0001 \
  --weight_decay 0.0001 \
  --beta1 0.9 \
  --beta2 0.999 \
  --temp 0.5 \
  --mode SHAM \
  --model vit_b_16 \
  --seed 42 \
  --num_workers 8 \
  --SHAM_mode reconstruction \
  --multi_view \
  --continue_training \
  --checkpoint_folder /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/output_dir/SHAM_vit_b_16_reconstruction_multi_view_decoder_7_layers
  #--warm_up_epochs 30 \
  #--sampling_frequency 0 \
  #  --continue_training \
  # --checkpoint_folder /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/output_dir/SHAM_vit_b_16_reconstruction_hard_negative_mining \
  
