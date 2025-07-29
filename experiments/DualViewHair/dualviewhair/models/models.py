"""
Clean model implementations for DualViewHair.

All model variants with consistent interfaces and clear separation of concerns.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from .components import (
    BaseEncoder, ResNetBackbone, EmbeddingHead, ProjectionHead,
    SpatialAttention, CrossViewAlignment
)


class BaselineEncoder(BaseEncoder):
    """Original DualViewHair encoder (clean implementation)."""
    
    def __init__(self, embedding_dim: int = 256, projection_dim: int = 128, pretrained: bool = True):
        super().__init__(embedding_dim, projection_dim)
        
        self.backbone = ResNetBackbone(pretrained=pretrained, extract_layers=['layer4'])
        self.embedding_head = EmbeddingHead(2048, embedding_dim)
        self.projection_head = ProjectionHead(embedding_dim, embedding_dim, projection_dim)
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        # Extract features
        features = self.backbone(x)
        pooled_features = features['pooled']  # [B, 2048]
        
        # Get embeddings
        embeddings = self.embedding_head(pooled_features)  # [B, embedding_dim]
        
        if return_embedding:
            return embeddings
        
        # Get projections for contrastive learning
        projections = self.projection_head(embeddings)  # [B, projection_dim]
        return projections


class MultiScaleEncoder(BaseEncoder):
    """Multi-scale encoder with attention for fine-grained details."""
    
    def __init__(
        self, 
        embedding_dim: int = 256, 
        projection_dim: int = 128, 
        pretrained: bool = True,
        use_attention: bool = True
    ):
        super().__init__(embedding_dim, projection_dim)
        
        self.use_attention = use_attention
        self.backbone = ResNetBackbone(
            pretrained=pretrained, 
            extract_layers=['layer2', 'layer3', 'layer4']
        )
        
        # Attention modules
        if use_attention:
            self.early_attention = SpatialAttention(512)   # layer2
            self.mid_attention = SpatialAttention(1024)    # layer3
        
        # Feature projections
        self.early_proj = nn.Linear(512, 64)    # Fine details
        self.mid_proj = nn.Linear(1024, 64)     # Patterns
        self.late_proj = nn.Linear(2048, 128)   # Global shape
        
        # Pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Final heads
        self.embedding_head = EmbeddingHead(256, embedding_dim)  # 64+64+128=256
        self.projection_head = ProjectionHead(embedding_dim, embedding_dim, projection_dim)
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        # Extract multi-scale features
        features = self.backbone(x)
        
        # Early features (fine-grained details)
        early_feat = features['layer2']  # [B, 512, 28, 28]
        if self.use_attention:
            early_feat = self.early_attention(early_feat)
        early_feat = self.pool(early_feat).flatten(1)
        early_proj = self.early_proj(early_feat)  # [B, 64]
        
        # Mid features (patterns)
        mid_feat = features['layer3']  # [B, 1024, 14, 14]
        if self.use_attention:
            mid_feat = self.mid_attention(mid_feat)
        mid_feat = self.pool(mid_feat).flatten(1)
        mid_proj = self.mid_proj(mid_feat)  # [B, 64]
        
        # Late features (global context)
        late_feat = features['pooled']  # [B, 2048]
        late_proj = self.late_proj(late_feat)  # [B, 128]
        
        # Fuse multi-scale features
        fused_features = torch.cat([early_proj, mid_proj, late_proj], dim=1)  # [B, 256]
        
        # Get embeddings
        embeddings = self.embedding_head(fused_features)
        
        if return_embedding:
            return embeddings
        
        # Get projections
        projections = self.projection_head(embeddings)
        return projections


class PartBasedEncoder(BaseEncoder):
    """Part-based encoder for semantic hair region learning."""
    
    def __init__(
        self, 
        embedding_dim: int = 256, 
        projection_dim: int = 128, 
        pretrained: bool = True,
        num_parts: int = 3
    ):
        super().__init__(embedding_dim, projection_dim)
        
        self.num_parts = num_parts
        self.backbone = ResNetBackbone(pretrained=pretrained, extract_layers=['layer4'])
        
        # Part-specific attention modules
        self.part_attentions = nn.ModuleList([
            SpatialAttention(2048) for _ in range(num_parts)
        ])
        
        # Part-specific embedding heads
        part_dim = embedding_dim // num_parts
        self.part_embeddings = nn.ModuleList([
            EmbeddingHead(2048, part_dim) for _ in range(num_parts)
        ])
        
        # Global fusion
        self.global_embedding = EmbeddingHead(embedding_dim, embedding_dim)
        self.projection_head = ProjectionHead(embedding_dim, embedding_dim, projection_dim)
        self.pool = nn.AdaptiveAvgPool2d(1)
    
    def forward(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        # Extract backbone features
        features = self.backbone(x)
        spatial_features = features['layer4']  # [B, 2048, 7, 7]
        
        # Apply part-specific attention and extract features
        part_features = []
        for i in range(self.num_parts):
            # Generate attention for this part
            attended_features = self.part_attentions[i](spatial_features)
            
            # Pool and get part-specific embedding
            pooled = self.pool(attended_features).flatten(1)  # [B, 2048]
            part_emb = self.part_embeddings[i](pooled)
            part_features.append(part_emb)
        
        # Combine part features
        combined_features = torch.cat(part_features, dim=1)  # [B, embedding_dim]
        
        # Global embedding
        embeddings = self.global_embedding(combined_features)
        
        if return_embedding:
            return embeddings
        
        # Get projections
        projections = self.projection_head(embeddings)
        return projections


class DualViewHairModel(nn.Module):
    """Main DualViewHair model with teacher-student architecture."""
    
    def __init__(
        self,
        encoder_type: str = "baseline",
        embedding_dim: int = 256,
        projection_dim: int = 128,
        momentum: float = 0.999,
        pretrained: bool = True,
        use_cross_alignment: bool = False,
        **encoder_kwargs
    ):
        super().__init__()
        
        self.momentum = momentum
        self.use_cross_alignment = use_cross_alignment
        
        # Select encoder class
        encoder_classes = {
            'baseline': BaselineEncoder,
            'multiscale': MultiScaleEncoder,
            'partbased': PartBasedEncoder
        }
        
        if encoder_type not in encoder_classes:
            raise ValueError(f"Unknown encoder type: {encoder_type}")
        
        encoder_class = encoder_classes[encoder_type]
        
        # Create student and teacher encoders
        self.student_encoder = encoder_class(
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            pretrained=pretrained,
            **encoder_kwargs
        )
        
        self.teacher_encoder = encoder_class(
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            pretrained=pretrained,
            **encoder_kwargs
        )
        
        # Cross-view alignment
        if use_cross_alignment:
            self.cross_alignment = CrossViewAlignment(embedding_dim)
        
        # Initialize teacher with student weights
        self._copy_student_to_teacher()
        
        # Freeze teacher parameters
        for param in self.teacher_encoder.parameters():
            param.requires_grad = False
    
    def _copy_student_to_teacher(self):
        """Copy student parameters to teacher."""
        for student_param, teacher_param in zip(
            self.student_encoder.parameters(),
            self.teacher_encoder.parameters()
        ):
            teacher_param.data.copy_(student_param.data)
    
    @torch.no_grad()
    def _update_teacher(self):
        """Update teacher parameters using exponential moving average."""
        for student_param, teacher_param in zip(
            self.student_encoder.parameters(),
            self.teacher_encoder.parameters()
        ):
            teacher_param.data.mul_(self.momentum).add_(
                student_param.data, alpha=1.0 - self.momentum
            )
    
    def forward(
        self,
        view_a: torch.Tensor,  # Hair regions (teacher input)
        view_b: torch.Tensor,  # Full images (student input)
        update_teacher: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through dual-view model.
        
        Args:
            view_a: Hair-only images [B, C, H, W]
            view_b: Full images [B, C, H, W]
            update_teacher: Whether to update teacher parameters
            
        Returns:
            Dictionary with model outputs
        """
        # Student forward pass
        student_proj = self.student_encoder(view_b, return_embedding=False)
        student_emb = self.student_encoder(view_b, return_embedding=True)
        
        # Teacher forward pass
        with torch.no_grad():
            teacher_proj = self.teacher_encoder(view_a, return_embedding=False)
            teacher_emb = self.teacher_encoder(view_a, return_embedding=True)
        
        outputs = {
            'student_projection': student_proj,
            'student_embedding': student_emb,
            'teacher_projection': teacher_proj,
            'teacher_embedding': teacher_emb
        }
        
        # Cross-view alignment
        if self.use_cross_alignment:
            aligned_teacher_emb, aligned_student_emb = self.cross_alignment(
                teacher_emb, student_emb
            )
            
            # Recompute projections with aligned embeddings
            aligned_teacher_proj = self.teacher_encoder.projection_head(aligned_teacher_emb)
            aligned_student_proj = self.student_encoder.projection_head(aligned_student_emb)
            
            outputs.update({
                'aligned_student_projection': aligned_student_proj,
                'aligned_student_embedding': aligned_student_emb,
                'aligned_teacher_projection': aligned_teacher_proj,
                'aligned_teacher_embedding': aligned_teacher_emb
            })
        
        # Update teacher parameters
        if update_teacher and self.training:
            self._update_teacher()
        
        return outputs
    
    def get_embeddings(self, images: torch.Tensor, use_teacher: bool = False) -> torch.Tensor:
        """Extract embeddings for retrieval."""
        encoder = self.teacher_encoder if use_teacher else self.student_encoder
        
        with torch.no_grad():
            embeddings = encoder(images, return_embedding=True)
        
        return embeddings
