python src/hair_retrieval.py \
    --ckpt_path "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/HairLearning/baselines/Siamese-Image-Modeling/checkpoint/checkpoint-199.pth" \
    --data_path "/path/to/data" \
    --embed_save_dir "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/HairLearning/baselines/Siamese-Image-Modeling/embeddings" \
    --batch_size 32 \
    --top_k 10 \
    --save_visualization \
    --num_queries 5 \
    --vis_save_dir "save/visualizations" \
    --random_seed 420 