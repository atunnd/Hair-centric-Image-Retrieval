#!/usr/bin/env python3
"""
Main training script for DualViewHair models.

Usage:
    python -m dualviewhair.train --config configs/baseline.yaml
    python -m dualviewhair.train --config configs/multiscale.yaml --resume checkpoints/latest.pth
"""

import argparse
import sys
from pathlib import Path

import torch

from .utils.config import load_config
from .data import create_data_loaders, get_dataset_stats
from .training.trainer import create_trainer


def main():
    parser = argparse.ArgumentParser(description="Train DualViewHair models")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size"
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of epochs"
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Override learning rate"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda"],
        help="Override device"
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loader workers"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config from {args.config}: {e}")
        sys.exit(1)
    
    # Override config values from command line
    if args.data_root:
        config.data.data_root = args.data_root
    if args.output_dir:
        config.training.output_dir = args.output_dir
    if args.batch_size:
        config.training.batch_size = args.batch_size
    if args.num_epochs:
        config.training.num_epochs = args.num_epochs
    if args.learning_rate:
        config.training.learning_rate = args.learning_rate
    
    # Set device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("=" * 60)
    print("DualViewHair Training")
    print("=" * 60)
    print(f"Config: {args.config}")
    print(f"Device: {device}")
    print(f"Data root: {config.data.data_root}")
    print(f"Output dir: {config.training.output_dir}")
    print(f"Model: {config.model.encoder_type}")
    print(f"Loss: {config.loss.loss_type}")
    print(f"Batch size: {config.training.batch_size}")
    print(f"Epochs: {config.training.num_epochs}")
    print(f"Learning rate: {config.training.learning_rate}")
    
    # Check data directory
    data_root = Path(config.data.data_root)
    if not data_root.exists():
        print(f"Error: Data directory not found: {data_root}")
        sys.exit(1)
    
    # Get dataset statistics
    try:
        dataset_stats = get_dataset_stats(config.data)
        print("\nDataset Statistics:")
        for split, stats in dataset_stats.items():
            if split != 'total_samples' and stats:
                print(f"  {split}: {stats['matching_pairs']} pairs")
        print(f"  Total: {dataset_stats['total_samples']} samples")
    except Exception as e:
        print(f"Warning: Could not get dataset stats: {e}")
    
    # Create data loaders
    try:
        train_loader, val_loader, test_loader = create_data_loaders(
            data_config=config.data,
            batch_size=config.training.batch_size,
            num_workers=args.num_workers
        )
        
        if train_loader is None:
            print("Error: No training data found")
            sys.exit(1)
        
        print(f"\nData loaders created:")
        print(f"  Train: {len(train_loader)} batches")
        if val_loader:
            print(f"  Validation: {len(val_loader)} batches")
        if test_loader:
            print(f"  Test: {len(test_loader)} batches")
            
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        sys.exit(1)
    
    # Create trainer
    try:
        trainer = create_trainer(
            model_config=config.model,
            loss_config=config.loss,
            training_config=config.training,
            train_loader=train_loader,
            val_loader=val_loader
        )
        print(f"\nTrainer created successfully")
        
        # Print model info
        total_params = sum(p.numel() for p in trainer.model.parameters())
        trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        print(f"Model parameters: {trainable_params:,} trainable / {total_params:,} total")
        
    except Exception as e:
        print(f"Error creating trainer: {e}")
        sys.exit(1)
    
    # Start training
    try:
        print("\n" + "=" * 60)
        print("Starting Training")
        print("=" * 60)
        
        trainer.train(resume_from=args.resume)
        
        print("\n" + "=" * 60)
        print("Training Completed Successfully!")
        print("=" * 60)
        print(f"Best validation loss: {trainer.best_val_loss:.4f}")
        print(f"Model saved to: {trainer.output_dir}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        print("Saving current checkpoint...")
        trainer.save_checkpoint(trainer.current_epoch, is_best=False)
        print(f"Checkpoint saved to: {trainer.output_dir}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
