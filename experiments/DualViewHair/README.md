# DualViewHair: Instance-Aware Hairstyle Embeddings

## Objective
Learn discriminative, instance-aware hairstyle embeddings from your existing segmented hair data using asymmetric dual-view training.

## Overview
This project implements a self-supervised learning approach for hairstyle representation learning:
- **View A (Teacher)**: Your segmented hair regions with minimal augmentation
- **View B (Student)**: Full images with strong augmentations

## 🚀 Quick Start (Simplified for Your Data)

**Your existing data structure**:
- Full images: `/path/to/celeb10k/0.jpg, 1.jpg, ...`
- Hair regions: `/path/to/hair_training_data/train/dummy/0_hair.png, 1_hair.png, ...`

### Step 1: Test Your Data
```bash
cd /mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/DualViewHair

# Test if everything is set up correctly
python scripts/test_setup.py
```

### Step 2: Install Dependencies
```bash
pip install torch torchvision Pillow numpy tensorboard
```

### Step 3: Start Training
```bash
# Quick test with data inspection only
python scripts/simple_train.py --inspect_only

# Start actual training
python scripts/simple_train.py --epochs 50 --batch_size 32
```

### Step 4: Monitor Training
```bash
tensorboard --logdir ./logs
```
Open http://localhost:6006 to view training progress.

## 📁 Simplified Structure

```
DualViewHair/
├── 📄 scripts/
│   ├── test_setup.py          # Test your data structure
│   └── simple_train.py        # Simple training script
├── 🧠 src/
│   ├── data/simple_dataloader.py  # Clean data loader for your data
│   └── models/dual_view_model.py   # Teacher-student model
└── 📊 checkpoints/            # Saved models
    └── logs/                  # TensorBoard logs
```

## Project Structure
```
DualViewHair/
├── data/                    # Dataset management
├── src/                     # Source code
│   ├── models/             # Model architectures
│   ├── data/               # Data processing
│   ├── training/           # Training logic
│   └── utils/              # Utility functions
├── configs/                # Configuration files
├── experiments/            # Experiment results
├── notebooks/              # Jupyter notebooks for analysis
└── scripts/                # Training and evaluation scripts
```

## Key Features
- Asymmetric dual-view training for hairstyle representation learning
- Face alignment and hair segmentation preprocessing
- Instance-aware contrastive learning
- Retrieval evaluation metrics

## Getting Started
1. Prepare your dataset in `data/raw/`
2. Configure training parameters in `configs/`
3. Run preprocessing: `python scripts/preprocess_data.py`
4. Start training: `python scripts/train.py`

## Requirements
- PyTorch >= 1.10
- OpenCV for image processing
- Face detection library (dlib/mediapipe)
- Hair segmentation model
