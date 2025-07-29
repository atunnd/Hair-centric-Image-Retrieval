"""
Configuration management for DualViewHair.

Provides a clean, type-safe configuration system using dataclasses.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
import yaml
from pathlib import Path


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str = "baseline"  # baseline, multiscale, partbased
    backbone: str = "resnet50"
    pretrained: bool = True
    embedding_dim: int = 256
    projection_dim: int = 128
    momentum: float = 0.999
    
    # Enhanced model specific
    use_attention: bool = False
    use_cross_alignment: bool = False
    num_parts: int = 3  # for part-based models


@dataclass
class DataConfig:
    """Data configuration."""
    full_images_dir: str = ""
    hair_images_dir: str = ""
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    
    # Augmentation settings
    hair_crop_margin: int = 16
    full_crop_margin: int = 32
    color_jitter: Dict[str, float] = field(default_factory=lambda: {
        "brightness": 0.1,
        "contrast": 0.1, 
        "saturation": 0.05,
        "hue": 0.02
    })


@dataclass
class LossConfig:
    """Loss function configuration."""
    name: str = "infonce"  # infonce, ntxent, hybrid
    temperature: float = 0.07
    
    # NT-Xent specific
    student_weight: float = 1.0
    teacher_weight: float = 0.5
    
    # Hybrid loss specific
    alignment_weight: float = 0.1
    consistency_weight: float = 0.05


@dataclass
class TrainingConfig:
    """Training configuration."""
    epochs: int = 50
    learning_rate: float = 0.001
    optimizer: str = "adam"
    scheduler: Optional[str] = None
    
    # Checkpointing
    save_every: int = 10
    checkpoint_dir: str = "./checkpoints"
    
    # Logging
    log_every: int = 100
    log_dir: str = "./logs"


@dataclass
class EvaluationConfig:
    """Evaluation configuration."""
    benchmark_path: str = ""
    database_root: str = ""
    query_root: str = ""
    
    # FAISS settings
    index_dir: str = "./faiss_indices"
    batch_size: int = 64
    
    # Metrics
    recall_ks: List[int] = field(default_factory=lambda: [10, 20, 50])
    map_ks: List[int] = field(default_factory=lambda: [10, 20, 50])


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # General settings
    device: str = "cuda"
    seed: int = 42
    experiment_name: str = "dualviewhair_experiment"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls(
            model=ModelConfig(**config_dict.get('model', {})),
            data=DataConfig(**config_dict.get('data', {})),
            loss=LossConfig(**config_dict.get('loss', {})),
            training=TrainingConfig(**config_dict.get('training', {})),
            evaluation=EvaluationConfig(**config_dict.get('evaluation', {})),
            **{k: v for k, v in config_dict.items() 
               if k not in ['model', 'data', 'loss', 'training', 'evaluation']}
        )
    
    def to_yaml(self, yaml_path: str) -> None:
        """Save configuration to YAML file."""
        config_dict = {
            'model': self.model.__dict__,
            'data': self.data.__dict__,
            'loss': self.loss.__dict__,
            'training': self.training.__dict__,
            'evaluation': self.evaluation.__dict__,
            'device': self.device,
            'seed': self.seed,
            'experiment_name': self.experiment_name
        }
        
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    
    def update_from_dict(self, updates: Dict[str, Any]) -> None:
        """Update configuration from dictionary."""
        for section, values in updates.items():
            if hasattr(self, section) and isinstance(values, dict):
                section_obj = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
            elif hasattr(self, section):
                setattr(self, section, values)
