
python knn_classification.py \
  --save_path classification_output_dir_celebA \
  --size 224 \
  --train_annotation /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/data/data_train_combination3.csv \
  --test_annotation /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/data/data_test_combination3.csv \
  --img_dir /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/hair_regions \
  --mode SHAM \
  --model vit_b_16 \
  --checkpoint_path /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/output_dir/SHAM_vit_b_16_reconstruction/model_ckpt_latest.pth \
  --seed 42 \
  --num_workers 8 \
  --device cuda:3 \
  --batch_size 256 \
  --eval_type linear_prob \
  --SHAM_mode reconstruction \


    