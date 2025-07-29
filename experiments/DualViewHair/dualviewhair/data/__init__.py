"""
Clean data loading and preprocessing for DualViewHair.

Unified data pipeline with proper augmentations for hair vs full image views.
"""

import os
from typing import Dict, List, Optional, Tuple, Callable, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

from ..utils.config import DataConfig


class DualViewHairDataset(Dataset):
    """Dataset for dual-view hair learning."""
    
    def __init__(
        self,
        data_config: DataConfig,
        split: str = "train",
        transform_hair: Optional[Callable] = None,
        transform_full: Optional[Callable] = None
    ):
        self.data_config = data_config
        self.split = split
        self.transform_hair = transform_hair
        self.transform_full = transform_full
        
        # Load data paths
        self.data_root = Path(data_config.data_root)
        self.hair_paths, self.full_paths = self._load_data_paths()
        
        print(f"Loaded {len(self.hair_paths)} {split} samples")
    
    def _load_data_paths(self) -> Tuple[List[str], List[str]]:
        """Load hair and full image paths."""
        hair_dir = self.data_root / "hair_regions" / self.split
        full_dir = self.data_root / "full_images" / self.split
        
        if not hair_dir.exists() or not full_dir.exists():
            raise ValueError(f"Data directories not found: {hair_dir}, {full_dir}")
        
        # Get matching pairs
        hair_paths = []
        full_paths = []
        
        for hair_path in sorted(hair_dir.glob("*.jpg")):
            # Find corresponding full image
            full_path = full_dir / hair_path.name
            if full_path.exists():
                hair_paths.append(str(hair_path))
                full_paths.append(str(full_path))
        
        if len(hair_paths) == 0:
            raise ValueError(f"No matching pairs found in {hair_dir} and {full_dir}")
        
        return hair_paths, full_paths
    
    def __len__(self) -> int:
        return len(self.hair_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load images
        hair_image = Image.open(self.hair_paths[idx]).convert('RGB')
        full_image = Image.open(self.full_paths[idx]).convert('RGB')
        
        # Apply transforms
        if self.transform_hair:
            hair_image = self.transform_hair(hair_image)
        if self.transform_full:
            full_image = self.transform_full(full_image)
        
        return {
            'hair_image': hair_image,
            'full_image': full_image,
            'hair_path': self.hair_paths[idx],
            'full_path': self.full_paths[idx]
        }


def create_transforms(data_config: DataConfig, split: str) -> Tuple[Callable, Callable]:
    """Create transforms for hair and full images."""
    
    # Base transforms
    base_transforms = [
        transforms.Resize((data_config.image_size, data_config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=data_config.normalize_mean,
            std=data_config.normalize_std
        )
    ]
    
    if split == "train":
        # Training augmentations for hair regions (more conservative)
        hair_transforms = transforms.Compose([
            transforms.Resize((data_config.image_size, data_config.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.1,
                contrast=0.1,
                saturation=0.1,
                hue=0.05
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=data_config.normalize_mean,
                std=data_config.normalize_std
            )
        ])
        
        # Training augmentations for full images (more aggressive)
        full_transforms = transforms.Compose([
            transforms.Resize((data_config.image_size, data_config.image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.2,
                    contrast=0.2,
                    saturation=0.2,
                    hue=0.1
                )
            ], p=0.8),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2),
            transforms.RandomApply([
                transforms.RandomRotation(degrees=10)
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=data_config.normalize_mean,
                std=data_config.normalize_std
            )
        ])
        
    else:
        # Validation/test transforms (no augmentation)
        hair_transforms = transforms.Compose(base_transforms)
        full_transforms = transforms.Compose(base_transforms)
    
    return hair_transforms, full_transforms


def create_data_loaders(
    data_config: DataConfig,
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Tuple[DataLoader, Optional[DataLoader], Optional[DataLoader]]:
    """Create train, validation, and test data loaders."""
    
    loaders = {}
    
    for split in ["train", "val", "test"]:
        # Check if split exists
        split_dir = Path(data_config.data_root) / "hair_regions" / split
        if not split_dir.exists():
            print(f"Skipping {split} split (directory not found)")
            loaders[split] = None
            continue
        
        # Create transforms
        hair_transforms, full_transforms = create_transforms(data_config, split)
        
        # Create dataset
        dataset = DualViewHairDataset(
            data_config=data_config,
            split=split,
            transform_hair=hair_transforms,
            transform_full=full_transforms
        )
        
        # Create data loader
        shuffle = (split == "train")
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=shuffle  # Drop last incomplete batch for training
        )
        
        loaders[split] = loader
    
    return loaders["train"], loaders["val"], loaders["test"]


class SimpleDataset(Dataset):
    """Simple dataset for existing preprocessed data."""
    
    def __init__(
        self,
        hair_paths: List[str],
        full_paths: List[str],
        transform_hair: Optional[Callable] = None,
        transform_full: Optional[Callable] = None
    ):
        assert len(hair_paths) == len(full_paths), "Mismatched number of images"
        
        self.hair_paths = hair_paths
        self.full_paths = full_paths
        self.transform_hair = transform_hair
        self.transform_full = transform_full
    
    def __len__(self) -> int:
        return len(self.hair_paths)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Load images
        hair_image = Image.open(self.hair_paths[idx]).convert('RGB')
        full_image = Image.open(self.full_paths[idx]).convert('RGB')
        
        # Apply transforms
        if self.transform_hair:
            hair_image = self.transform_hair(hair_image)
        if self.transform_full:
            full_image = self.transform_full(full_image)
        
        return {
            'hair_image': hair_image,
            'full_image': full_image,
            'idx': idx
        }


def create_simple_loader(
    hair_paths: List[str],
    full_paths: List[str],
    data_config: DataConfig,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 4
) -> DataLoader:
    """Create simple data loader from path lists."""
    
    # Create transforms (no augmentation)
    hair_transforms, full_transforms = create_transforms(data_config, "test")
    
    # Create dataset
    dataset = SimpleDataset(
        hair_paths=hair_paths,
        full_paths=full_paths,
        transform_hair=hair_transforms,
        transform_full=full_transforms
    )
    
    # Create data loader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return loader


def get_dataset_stats(data_config: DataConfig) -> Dict[str, Any]:
    """Get statistics about the dataset."""
    data_root = Path(data_config.data_root)
    
    stats = {}
    total_samples = 0
    
    for split in ["train", "val", "test"]:
        hair_dir = data_root / "hair_regions" / split
        full_dir = data_root / "full_images" / split
        
        if hair_dir.exists() and full_dir.exists():
            hair_files = list(hair_dir.glob("*.jpg"))
            full_files = list(full_dir.glob("*.jpg"))
            
            # Count matching pairs
            matching_pairs = 0
            for hair_file in hair_files:
                full_file = full_dir / hair_file.name
                if full_file.exists():
                    matching_pairs += 1
            
            stats[split] = {
                'hair_images': len(hair_files),
                'full_images': len(full_files),
                'matching_pairs': matching_pairs
            }
            total_samples += matching_pairs
        else:
            stats[split] = None
    
    stats['total_samples'] = total_samples
    return stats
