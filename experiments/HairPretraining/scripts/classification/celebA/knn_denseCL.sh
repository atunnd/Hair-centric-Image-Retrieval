
python knn_classification.py \
  --save_path classification_output_dir_celebA \
  --size 224 \
  --train_annotation /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/data/data_train_combination3.csv \
  --test_annotation /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/data/data_test_combination3.csv \
  --img_dir /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/hair_regions \
  --mode DenseCL \
  --model resnet50 \
  --checkpoint_path /datastore/dragonzakura/QuocAnh/Composed-Image-Retrieval/experiments/HairPretraining/output_dir/DenseCL_resnet50/model_ckpt_latest.pth \
  --seed 42 \
  --num_workers 8 \
  --device cuda \
  --batch_size 256 \
  --eval_type linear_prob


    