python src/face_retrieval.py \
    --ckpt_path "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/weights/face_encoder/Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth" \
    --data_path "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/data/" \
    --embed_save_dir "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/save/embeddings" \
    --batch_size 32 \
    --top_k 10 \
    --save_visualization \
    --num_queries 5 \
    --vis_save_dir "save/visualizations" \
    --random_seed 421