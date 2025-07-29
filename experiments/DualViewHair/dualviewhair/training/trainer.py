"""
Unified training framework for DualViewHair models.

Clean, configurable trainer supporting all model variants and loss functions.
"""

import os
import json
import time
from typing import Dict, Optional, Tuple, Any
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

from ..utils.config import TrainingConfig, ModelConfig, LossConfig
from ..models.models import DualViewHairModel
from ..losses import create_loss_function


class Trainer:
    """Unified trainer for all DualViewHair model variants."""
    
    def __init__(
        self,
        model: DualViewHairModel,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TrainingConfig()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup loss function
        self.criterion = create_loss_function(self.config.loss)
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []
        
        # Setup output directory
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        self._save_config()
    
    def _create_optimizer(self) -> Optimizer:
        """Create optimizer from config."""
        if self.config.optimizer == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adamw":
            return torch.optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                momentum=0.9
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
    
    def _create_scheduler(self) -> Optional[_LRScheduler]:
        """Create learning rate scheduler from config."""
        if self.config.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs
            )
        elif self.config.scheduler == "step":
            return torch.optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=self.config.num_epochs // 3,
                gamma=0.1
            )
        elif self.config.scheduler == "warmup_cosine":
            # Simple warmup + cosine implementation
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.num_epochs - self.config.warmup_epochs
            )
        else:
            return None
    
    def _save_config(self):
        """Save training configuration."""
        config_dict = {
            'model_config': self.config.model.__dict__,
            'training_config': self.config.__dict__,
            'loss_config': self.config.loss.__dict__
        }
        
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        loss_components = {}
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move data to device
            view_a = batch['hair_image'].to(self.device)  # Hair regions
            view_b = batch['full_image'].to(self.device)  # Full images
            
            # Forward pass
            outputs = self.model(view_a, view_b)
            
            # Compute loss
            loss_dict = self.criterion(outputs)
            loss = loss_dict['loss']
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.grad_clip_norm
                )
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item() if torch.is_tensor(value) else value
            
            self.global_step += 1
            
            # Log progress
            if batch_idx % self.config.log_interval == 0:
                avg_loss = total_loss / (batch_idx + 1)
                print(f"Epoch {self.current_epoch}, Batch {batch_idx}/{num_batches}, "
                      f"Loss: {avg_loss:.4f}")
        
        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        # Update scheduler
        if self.scheduler is not None:
            self.scheduler.step()
        
        return {
            'avg_loss': avg_loss,
            **loss_components
        }
    
    @torch.no_grad()
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        
        total_loss = 0.0
        loss_components = {}
        num_batches = len(self.val_loader)
        
        for batch in self.val_loader:
            # Move data to device
            view_a = batch['hair_image'].to(self.device)
            view_b = batch['full_image'].to(self.device)
            
            # Forward pass
            outputs = self.model(view_a, view_b, update_teacher=False)
            
            # Compute loss
            loss_dict = self.criterion(outputs)
            loss = loss_dict['loss']
            
            # Update statistics
            total_loss += loss.item()
            for key, value in loss_dict.items():
                if key not in loss_components:
                    loss_components[key] = 0.0
                loss_components[key] += value.item() if torch.is_tensor(value) else value
        
        # Average losses
        avg_loss = total_loss / num_batches
        for key in loss_components:
            loss_components[key] /= num_batches
        
        return {
            'avg_loss': avg_loss,
            **loss_components
        }
    
    def save_checkpoint(self, epoch: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'config': self.config.__dict__
        }
        
        # Save regular checkpoint
        checkpoint_path = self.output_dir / f'checkpoint_epoch_{epoch:03d}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best checkpoint
        if is_best:
            best_path = self.output_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
        
        # Save latest checkpoint
        latest_path = self.output_dir / 'latest_checkpoint.pth'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
    
    def train(self, resume_from: Optional[str] = None):
        """Main training loop."""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model: {self.model.__class__.__name__}")
        print(f"Loss: {self.criterion.__class__.__name__}")
        print(f"Optimizer: {self.optimizer.__class__.__name__}")
        
        # Resume from checkpoint if specified
        if resume_from:
            self.load_checkpoint(resume_from)
            print(f"Resumed training from epoch {self.current_epoch}")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.num_epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Training
            train_metrics = self.train_epoch()
            self.train_losses.append(train_metrics['avg_loss'])
            
            # Validation
            val_metrics = self.validate_epoch()
            if val_metrics:
                self.val_losses.append(val_metrics['avg_loss'])
                current_val_loss = val_metrics['avg_loss']
                is_best = current_val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = current_val_loss
            else:
                is_best = False
            
            # Logging
            epoch_time = time.time() - epoch_start_time
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs} completed in {epoch_time:.2f}s")
            print(f"Train Loss: {train_metrics['avg_loss']:.4f}")
            if val_metrics:
                print(f"Val Loss: {val_metrics['avg_loss']:.4f} {'(Best!)' if is_best else ''}")
            
            # Save checkpoint
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch + 1, is_best)
        
        # Final save
        self.save_checkpoint(self.config.num_epochs, is_best=False)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f}s")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
    
    def get_model_for_inference(self) -> DualViewHairModel:
        """Get model ready for inference."""
        self.model.eval()
        return self.model


def create_trainer(
    model_config: ModelConfig,
    loss_config: LossConfig,
    training_config: TrainingConfig,
    train_loader: DataLoader,
    val_loader: Optional[DataLoader] = None
) -> Trainer:
    """Factory function to create trainer from configs."""
    
    # Create model
    model = DualViewHairModel(
        encoder_type=model_config.encoder_type,
        embedding_dim=model_config.embedding_dim,
        projection_dim=model_config.projection_dim,
        momentum=model_config.momentum,
        pretrained=model_config.pretrained,
        use_cross_alignment=model_config.use_cross_alignment
    )
    
    # Set loss config in training config
    training_config.loss = loss_config
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config
    )
    
    return trainer
