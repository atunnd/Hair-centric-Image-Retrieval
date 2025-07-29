"""
Base model components for DualViewHair.

Provides clean, reusable building blocks for all model variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from abc import ABC, abstractmethod
from typing import Dict, Tuple, Optional


class BaseEncoder(nn.Module, ABC):
    """Abstract base class for all encoders."""
    
    def __init__(self, embedding_dim: int = 256, projection_dim: int = 128):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
    
    @abstractmethod
    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """Forward pass through encoder."""
        pass


class SpatialAttention(nn.Module):
    """Spatial attention module for focusing on relevant regions."""
    
    def __init__(self, in_channels: int, reduction: int = 8):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attention_map = self.attention(x)  # [B, 1, H, W]
        return x * attention_map


class CrossViewAlignment(nn.Module):
    """Cross-view attention for aligning features between different views."""
    
    def __init__(self, embedding_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hair_to_full_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.full_to_hair_attention = nn.MultiheadAttention(
            embedding_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
    
    def forward(self, hair_features: torch.Tensor, full_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Align features between hair-only and full image views.
        
        Args:
            hair_features: [B, D] hair-only features
            full_features: [B, D] full image features
            
        Returns:
            Tuple of aligned (hair_features, full_features)
        """
        # Add sequence dimension for attention
        hair_seq = hair_features.unsqueeze(1)  # [B, 1, D]
        full_seq = full_features.unsqueeze(1)  # [B, 1, D]
        
        # Cross-attention: hair queries full
        aligned_hair, _ = self.hair_to_full_attention(hair_seq, full_seq, full_seq)
        aligned_hair = self.ln1(aligned_hair.squeeze(1) + hair_features)
        
        # Cross-attention: full queries hair
        aligned_full, _ = self.full_to_hair_attention(full_seq, hair_seq, hair_seq)
        aligned_full = self.ln2(aligned_full.squeeze(1) + full_features)
        
        return aligned_hair, aligned_full


class ResNetBackbone(nn.Module):
    """Clean ResNet backbone with configurable layer extraction."""
    
    def __init__(self, pretrained: bool = True, extract_layers: Optional[list] = None):
        super().__init__()
        
        # Load ResNet50
        weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        resnet = resnet50(weights=weights)
        
        # Extract layers
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.avgpool = resnet.avgpool
        
        # Define layer output channels
        self.layer_channels = {
            'layer1': 256,
            'layer2': 512,
            'layer3': 1024,
            'layer4': 2048
        }
        
        self.extract_layers = extract_layers or ['layer4']
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with optional multi-layer extraction.
        
        Returns:
            Dictionary mapping layer names to features
        """
        features = {}
        
        # Early layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # ResNet layers
        x = self.layer1(x)
        if 'layer1' in self.extract_layers:
            features['layer1'] = x
            
        x = self.layer2(x)
        if 'layer2' in self.extract_layers:
            features['layer2'] = x
            
        x = self.layer3(x)
        if 'layer3' in self.extract_layers:
            features['layer3'] = x
            
        x = self.layer4(x)
        if 'layer4' in self.extract_layers:
            features['layer4'] = x
        
        # Global pooling
        x = self.avgpool(x)
        features['pooled'] = x.flatten(1)
        
        return features


class ProjectionHead(nn.Module):
    """Configurable projection head for contrastive learning."""
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        
        layers = []
        current_dim = input_dim
        
        for i in range(num_layers):
            if i == num_layers - 1:
                # Final layer
                layers.extend([
                    nn.Linear(current_dim, output_dim),
                    nn.BatchNorm1d(output_dim)
                ])
            else:
                # Hidden layers
                layers.extend([
                    nn.Linear(current_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(inplace=True)
                ])
                current_dim = hidden_dim
        
        self.projection = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projection(x)


class EmbeddingHead(nn.Module):
    """Embedding head for feature extraction."""
    
    def __init__(self, input_dim: int, embedding_dim: int, dropout: float = 0.1):
        super().__init__()
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)
