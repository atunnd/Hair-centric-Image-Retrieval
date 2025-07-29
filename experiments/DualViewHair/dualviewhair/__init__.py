"""
DualViewHair: Clean implementation for hairstyle instance representation learning.

A PyTorch-based framework for dual-view contrastive learning on hairstyle images.
"""

__version__ = "2.0.0"
__author__ = "DualViewHair Team"

# Main components
from .models.models import DualViewHairModel, BaselineEncoder, MultiScaleEncoder, PartBasedEncoder
from .losses import InfoNCELoss, NTXentLoss, AsymmetricNTXentLoss, HybridLoss, create_loss_function
from .training.trainer import Trainer, create_trainer
from .data import DualViewHairDataset, create_data_loaders, create_simple_loader
from .utils.config import (
    ModelConfig, DataConfig, LossConfig, TrainingConfig, EvaluationConfig,
    load_config, save_config
)

# Package metadata
__all__ = [
    # Models
    "DualViewHairModel",
    "BaselineEncoder", 
    "MultiScaleEncoder",
    "PartBasedEncoder",
    
    # Losses
    "InfoNCELoss",
    "NTXentLoss", 
    "AsymmetricNTXentLoss",
    "HybridLoss",
    "create_loss_function",
    
    # Training
    "Trainer",
    "create_trainer",
    
    # Data
    "DualViewHairDataset",
    "create_data_loaders",
    "create_simple_loader",
    
    # Configuration
    "ModelConfig",
    "DataConfig", 
    "LossConfig",
    "TrainingConfig",
    "EvaluationConfig",
    "load_config",
    "save_config",
]
