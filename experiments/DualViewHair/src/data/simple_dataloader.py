"""Simple data loader for DualViewHair with existing segmented hair regions."""

import random
from pathlib import Path
from typing import Tuple, List, Dict

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image


class HairDataset(Dataset):
    """Dataset for dual-view hairstyle learning."""
    
    def __init__(self, full_dir: str, hair_dir: str, image_size: int = 224):
        self.image_size = image_size
        
        # Find matching pairs
        full_path = Path(full_dir)
        hair_path = Path(hair_dir)
        
        pairs = []
        for full_img in full_path.glob("*.jpg"):
            hair_img = hair_path / f"{full_img.stem}_hair.png"
            if hair_img.exists():
                pairs.append({'id': full_img.stem, 'full': str(full_img), 'hair': str(hair_img)})
        
        # Use all pairs for training
        self.pairs = pairs
        
        # Transforms - always use training augmentations
        self.hair_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        
        self.full_transform = transforms.Compose([
            transforms.Resize((image_size + 32, image_size + 32)),
            transforms.RandomCrop((image_size, image_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        pair = self.pairs[idx]
        
        full_img = Image.open(pair['full']).convert('RGB')
        hair_img = Image.open(pair['hair']).convert('RGB')
        return {
            'view_a': self.hair_transform(hair_img),  # Teacher
            'view_b': self.full_transform(full_img),  # Student
            'image_id': pair['id']
        }
    

def create_dataloader(full_dir: str, hair_dir: str, batch_size: int = 32, 
                     image_size: int = 224, num_workers: int = 4):
    """Create training dataloader."""
    
    dataset = HairDataset(full_dir, hair_dir, image_size)
    
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    
    return loader
