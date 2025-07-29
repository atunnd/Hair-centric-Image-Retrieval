"""
Enhanced DualViewHair model with fine-grained detail capture.

Implements multi-scale attention and part-based learning for better hairstyle details.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Tuple, Dict, Optional


class SpatialAttention(nn.Module):
    """Spatial attention module to focus on hair regions in full images."""
    
    def __init__(self, in_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels // 8, 1)
        self.conv2 = nn.Conv2d(in_channels // 8, 1, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: [B, C, H, W]
        attention = self.conv1(x)
        attention = F.relu(attention)
        attention = self.conv2(attention)
        attention = self.sigmoid(attention)  # [B, 1, H, W]
        
        return x * attention


class MultiScaleHairEncoder(nn.Module):
    """
    Multi-scale encoder for capturing fine-grained hairstyle details.
    
    Combines global features with local hair-specific patterns.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        embedding_dim: int = 256,
        projection_dim: int = 128,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        self.use_attention = use_attention
        
        # Backbone network
        if backbone == 'resnet50':
            if pretrained:
                weights = ResNet50_Weights.IMAGENET1K_V2
            else:
                weights = None
            backbone_model = resnet50(weights=weights)
            
            # Extract feature layers for multi-scale processing
            backbone_layers = list(backbone_model.children())
            self.backbone_early = nn.Sequential(*backbone_layers[:-4])  # Up to layer2: conv1, bn1, relu, maxpool, layer1, layer2
            self.backbone_mid = backbone_layers[-4]                     # layer3 only
            self.backbone_late = backbone_layers[-3]                    # layer4 only
            self.backbone_pool = backbone_layers[-2]                    # avgpool
            
            # Feature dimensions at different scales
            self.early_dim = 512   
            self.mid_dim = 1024    
            self.late_dim = 2048   
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Spatial attention for focusing on hair regions
        if self.use_attention:
            self.early_attention = SpatialAttention(self.early_dim)
            self.mid_attention = SpatialAttention(self.mid_dim)
        
        # Multi-scale feature fusion
        self.early_pool = nn.AdaptiveAvgPool2d(1)
        self.mid_pool = nn.AdaptiveAvgPool2d(1)
        
        # Feature projection layers
        self.early_proj = nn.Linear(self.early_dim, embedding_dim // 4)
        self.mid_proj = nn.Linear(self.mid_dim, embedding_dim // 4)
        self.late_proj = nn.Linear(self.late_dim, embedding_dim // 2)
        
        # Final embedding head
        self.embedding_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Projection head for contrastive learning
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """Multi-scale forward pass with attention."""
        
        # Extract features at different scales from the same input
        # Early features (fine-grained details) - up to layer2
        early_feat = self.backbone_early(x)  # [B, 512, 28, 28]
        if self.use_attention:
            early_feat = self.early_attention(early_feat)
        early_pooled = self.early_pool(early_feat).flatten(1)  # [B, 512]
        early_projected = self.early_proj(early_pooled)  # [B, 64]
        
        # Mid-level features (hair patterns) - layer3
        mid_feat = self.backbone_mid(early_feat)  # [B, 1024, 14, 14]
        if self.use_attention:
            mid_feat = self.mid_attention(mid_feat)
        mid_pooled = self.mid_pool(mid_feat).flatten(1)  # [B, 1024]
        mid_projected = self.mid_proj(mid_pooled)  # [B, 64]
        
        # Late features (global context) - layer4
        late_feat = self.backbone_late(mid_feat)  # [B, 2048, 7, 7]
        late_pooled = self.backbone_pool(late_feat).flatten(1)  # [B, 2048]
        late_projected = self.late_proj(late_pooled)  # [B, 128]
        
        # Fuse multi-scale features
        fused_features = torch.cat([early_projected, mid_projected, late_projected], dim=1)  # [B, 256]
        
        # Final embedding
        embeddings = self.embedding_head(fused_features)
        
        if return_embedding:
            return embeddings
        
        # Project for contrastive learning
        projections = self.projection_head(embeddings)
        
        return projections


class PartBasedHairEncoder(nn.Module):
    """
    Part-based encoder that learns different hair regions separately.
    
    Divides hair into semantic parts (top, sides, back) for fine-grained learning.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        embedding_dim: int = 256,
        projection_dim: int = 128,
        num_parts: int = 3
    ):
        super().__init__()
        
        self.num_parts = num_parts
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        
        # Shared backbone
        if backbone == 'resnet50':
            if pretrained:
                weights = ResNet50_Weights.IMAGENET1K_V2
            else:
                weights = None
            backbone_model = resnet50(weights=weights)
            self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Part-specific attention modules
        self.part_attentions = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(backbone_dim, backbone_dim // 8, 1),
                nn.ReLU(),
                nn.Conv2d(backbone_dim // 8, 1, 1),
                nn.Sigmoid()
            ) for _ in range(num_parts)
        ])
        
        # Part-specific embedding heads
        self.part_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(backbone_dim, embedding_dim // num_parts),
                nn.BatchNorm1d(embedding_dim // num_parts),
                nn.ReLU(inplace=True)
            ) for _ in range(num_parts)
        ])
        
        # Global embedding fusion
        self.global_embedding = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        # Projection head
        self.projection_head = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.Linear(embedding_dim, projection_dim),
            nn.BatchNorm1d(projection_dim)
        )
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """Part-based forward pass."""
        
        # Extract backbone features
        features = self.backbone(x)  # [B, 2048, 7, 7]
        
        # Apply part-specific attention and extract features
        part_features = []
        for i in range(self.num_parts):
            # Generate attention map for this part
            attention = self.part_attentions[i](features)  # [B, 1, 7, 7]
            
            # Apply attention and pool
            attended_features = features * attention
            pooled_features = F.adaptive_avg_pool2d(attended_features, 1).flatten(1)  # [B, 2048]
            
            # Get part-specific embedding
            part_emb = self.part_embeddings[i](pooled_features)  # [B, 256//num_parts]
            part_features.append(part_emb)
        
        # Concatenate part features
        combined_features = torch.cat(part_features, dim=1)  # [B, 256]
        
        # Global embedding
        embeddings = self.global_embedding(combined_features)
        
        if return_embedding:
            return embeddings
        
        # Project for contrastive learning
        projections = self.projection_head(embeddings)
        
        return projections


class CrossViewAlignment(nn.Module):
    """
    Cross-view alignment module to better align hair-only and full image features.
    """
    
    def __init__(self, embedding_dim: int = 256):
        super().__init__()
        
        # Cross-attention between hair and full image features
        self.hair_to_full_attention = nn.MultiheadAttention(
            embedding_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        self.full_to_hair_attention = nn.MultiheadAttention(
            embedding_dim, num_heads=8, dropout=0.1, batch_first=True
        )
        
        # Layer normalization
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)
        
    def forward(self, hair_features: torch.Tensor, full_features: torch.Tensor):
        """
        Align features between hair-only and full image views.
        
        Args:
            hair_features: [B, D] hair-only features
            full_features: [B, D] full image features
            
        Returns:
            Aligned hair and full features
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
