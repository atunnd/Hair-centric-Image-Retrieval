# DualViewHair: Quick Start Guide

This guide will help you get started with training the DualViewHair model using your existing CelebA-10K and hair region data.

## Prerequisites

1. **Python Environment**: Python 3.8+
2. **Dependencies**: Install required packages
```bash
cd /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/DualViewHair
pip install -r requirements.txt
```

## Your Data Structure

Based on your setup:
- **Full images**: `/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/celeb10k`
  - Format: `0.jpg`, `1.jpg`, `2.jpg`, ...
- **Hair regions**: `/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/hair_representation/hair_training_data/train/dummy`
  - Format: `0_hair.png`, `1_hair.png`, `2_hair.png`, ...

## Step 1: Inspect Your Data

First, verify that your data structure is correctly set up:

```bash
python scripts/inspect_data.py \\
  --full_images_dir "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/celeb10k" \\
  --hair_images_dir "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/hair_representation/hair_training_data/train/dummy" \\
  --show_samples 10
```

This script will:
- ✅ Check if both directories exist
- 🔗 Find matching pairs between full images and hair regions
- 📋 Show sample matched pairs
- ❌ Report any unmatched files
- 📊 Analyze file sizes

Expected output:
```
=== Data Structure Inspection ===
✅ Both directories exist
📁 Directory contents:
  Full images found: 10000
  Hair images found: 10000
🔗 Matching analysis:
  Matched pairs: 10000
  Unmatched full images: 0
  Unmatched hair images: 0
```

## Step 2: Configure Training

The configuration file `configs/default.yaml` has been pre-configured for your data structure:

```yaml
# Data configuration
data:
  full_images_dir: "/path/to/celeb10k"
  hair_images_dir: "/path/to/hair_training_data/train/dummy"
  full_image_ext: ".jpg"
  hair_image_suffix: "_hair"
  hair_image_ext: ".png"
  image_size: 224
  batch_size: 32
  num_workers: 4
```

You can modify other settings like:
- `model.backbone`: Architecture (resnet50, etc.)
- `training.learning_rate`: Learning rate
- `training.epochs`: Number of epochs
- `training.batch_size`: Batch size

## Step 3: Start Training

### Option A: Basic Training
```bash
python scripts/train_existing_data.py \\
  --config configs/default.yaml \\
  --experiment_name "dualviewhair_celeb10k_v1"
```

### Option B: Custom Settings
```bash
python scripts/train_existing_data.py \\
  --config configs/default.yaml \\
  --full_images_dir "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/celeb10k" \\
  --hair_images_dir "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/hair_representation/hair_training_data/train/dummy" \\
  --epochs 50 \\
  --batch_size 64 \\
  --learning_rate 1e-3 \\
  --experiment_name "dualviewhair_celeb10k_fast" \\
  --save_splits
```

### Training Arguments:
- `--config`: Configuration file path
- `--full_images_dir`: Path to full images directory
- `--hair_images_dir`: Path to hair regions directory
- `--epochs`: Number of training epochs
- `--batch_size`: Batch size
- `--learning_rate`: Learning rate
- `--experiment_name`: Name for this experiment
- `--save_splits`: Save dataset splits to file
- `--resume`: Resume from checkpoint
- `--device`: Device (cuda/cpu/auto)

## Step 4: Monitor Training

Training progress is logged to TensorBoard:

```bash
# In a new terminal
tensorboard --logdir experiments/dualviewhair_celeb10k_v1/logs
```

Open http://localhost:6006 to view:
- Training/validation loss curves
- Learning rate schedule
- Retrieval metrics (mAP, Recall@k)

## Step 5: Evaluate Results

After training, the script automatically evaluates on the test set and saves results to:
- `experiments/{experiment_name}/test_results.yaml`
- `experiments/{experiment_name}/checkpoints/best_model.pth`

## Training Process Details

### Dual-View Architecture:
1. **View A (Teacher)**: Hair-only images with minimal augmentation
2. **View B (Student)**: Full images with strong augmentations
3. **Contrastive Learning**: Teacher-student knowledge distillation

### Data Flow:
```
Full Image (0.jpg) → View B (Student) → Student Encoder → Projections
Hair Region (0_hair.png) → View A (Teacher) → Teacher Encoder → Projections
                                    ↓
                            Contrastive Loss
```

### Expected Training Output:
```
=== DualViewHair Training (Existing Data) ===
Using device: cuda
GPU: NVIDIA RTX 4090
Created model with:
  Backbone: resnet50
  Embedding dim: 256
  Projection dim: 128
  Total parameters: 25,557,032
  Trainable parameters: 25,557,032

Created data loaders:
  Train: 250 batches (8000 samples)
  Val: 32 batches (1000 samples)  
  Test: 32 batches (1000 samples)

Starting training for 100 epochs...
Epoch 1/100 (45.2s)
  Train Loss: 2.3456
  Val Loss: 2.1234
...
```

## Troubleshooting

### Common Issues:

1. **"No matching pairs found"**
   - Check file naming convention
   - Verify directory paths
   - Run `scripts/inspect_data.py` for details

2. **CUDA out of memory**
   - Reduce batch size: `--batch_size 16`
   - Reduce image size in config: `image_size: 224 → 192`

3. **Slow training**
   - Increase num_workers: `num_workers: 8`
   - Use mixed precision: Set `hardware.mixed_precision: true` in config

4. **Poor convergence**
   - Adjust learning rate: `--learning_rate 1e-4`
   - Try different backbone: `model.backbone: "resnet34"`
   - Increase momentum: `model.momentum: 0.999`

## Advanced Usage

### Custom Configuration:
Create a new config file based on `configs/default.yaml`:

```bash
cp configs/default.yaml configs/my_experiment.yaml
# Edit configs/my_experiment.yaml
python scripts/train_existing_data.py --config configs/my_experiment.yaml
```

### Resume Training:
```bash
python scripts/train_existing_data.py \\
  --config configs/default.yaml \\
  --resume experiments/dualviewhair_celeb10k_v1/checkpoints/best_model.pth
```

### Save Dataset Splits:
```bash
python scripts/train_existing_data.py \\
  --config configs/default.yaml \\
  --save_splits \\
  --experiment_name "split_analysis"
```

## Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes, architectures
2. **Analyze results**: Use TensorBoard to understand training dynamics  
3. **Evaluate retrieval**: Test hairstyle-based image retrieval on your dataset
4. **Fine-tune**: Adjust augmentations and loss functions for better performance

For more details, see the full documentation in `README.md`.
