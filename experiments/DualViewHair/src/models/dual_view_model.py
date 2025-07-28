"""
DualViewHair model architectures for instance-aware hairstyle embeddings.

Implements asymmetric teacher-student architecture with contrastive learning.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from typing import Tuple, Dict, Optional

class HairstyleEncoder(nn.Module):
    """
    Encoder network for hairstyle feature extraction.
    
    Uses ResNet-50 backbone with custom projection head for hairstyle embeddings.
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        embedding_dim: int = 256,
        projection_dim: int = 128
    ):
        """
        Initialize hairstyle encoder.
        
        Args:
            backbone: Backbone architecture name
            pretrained: Whether to use pretrained weights
            embedding_dim: Dimension of intermediate embeddings
            projection_dim: Dimension of final projections for contrastive learning
        """
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.projection_dim = projection_dim
        
        # Backbone network
        if backbone == 'resnet50':
            if pretrained:
                weights = ResNet50_Weights.IMAGENET1K_V2
            else:
                weights = None
            backbone_model = resnet50(weights=weights)
            
            # Remove final classification layer
            self.backbone = nn.Sequential(*list(backbone_model.children())[:-1])
            backbone_dim = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Feature embedding head
        self.embedding_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(backbone_dim, embedding_dim),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True)
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
        """
        Forward pass through encoder.
        
        Args:
            x: Input images [B, C, H, W]
            return_embedding: Whether to return intermediate embeddings
            
        Returns:
            Projected features [B, projection_dim] or embeddings [B, embedding_dim]
        """
        # Extract backbone features
        features = self.backbone(x)  # [B, backbone_dim, 1, 1]
        
        # Get embeddings
        embeddings = self.embedding_head(features)  # [B, embedding_dim]
        
        if return_embedding:
            return embeddings
        
        # Project for contrastive learning
        projections = self.projection_head(embeddings)  # [B, projection_dim]
        
        return projections


class DualViewHairModel(nn.Module):
    """
    Dual-view model for asymmetric hairstyle representation learning.
    
    Implements teacher-student architecture where:
    - Teacher processes hair-only images (View A)
    - Student processes full images (View B)
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        embedding_dim: int = 256,
        projection_dim: int = 128,
        momentum: float = 0.999,
        temperature: float = 0.07
    ):
        """
        Initialize dual-view model.
        
        Args:
            backbone: Backbone architecture
            pretrained: Use pretrained weights
            embedding_dim: Embedding dimension
            projection_dim: Projection dimension
            momentum: Momentum for teacher updates
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        
        self.momentum = momentum
        self.temperature = temperature
        
        # Student encoder (processes full images - View B)
        self.student_encoder = HairstyleEncoder(
            backbone=backbone,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim
        )
        
        # Teacher encoder (processes hair-only images - View A)
        self.teacher_encoder = HairstyleEncoder(
            backbone=backbone,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim
        )
        
        # Initialize teacher with student weights
        self._copy_student_to_teacher()
        
        # Freeze teacher parameters (will be updated via momentum)
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
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        update_teacher: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through dual-view model.
        
        Args:
            view_a: Hair-only images (teacher input) [B, C, H, W]
            view_b: Full images (student input) [B, C, H, W]
            update_teacher: Whether to update teacher parameters
            
        Returns:
            Dictionary with teacher and student outputs
        """
        # Student forward pass (full images)
        student_proj = self.student_encoder(view_b, return_embedding=False)
        student_emb = self.student_encoder(view_b, return_embedding=True)
        
        # Teacher forward pass (hair-only images)
        with torch.no_grad():
            teacher_proj = self.teacher_encoder(view_a, return_embedding=False)
            teacher_emb = self.teacher_encoder(view_a, return_embedding=True)
        
        # Update teacher parameters
        if update_teacher and self.training:
            self._update_teacher()
        
        return {
            'student_projection': student_proj,
            'student_embedding': student_emb,
            'teacher_projection': teacher_proj,
            'teacher_embedding': teacher_emb
        }
    
    def get_embeddings(self, images: torch.Tensor, use_teacher: bool = False) -> torch.Tensor:
        """
        Extract embeddings for retrieval.
        
        Args:
            images: Input images [B, C, H, W]
            use_teacher: Whether to use teacher encoder
            
        Returns:
            Embeddings [B, embedding_dim]
        """
        encoder = self.teacher_encoder if use_teacher else self.student_encoder
        
        with torch.no_grad():
            embeddings = encoder(images, return_embedding=True)
        
        return embeddings


class ContrastiveLoss(nn.Module):
    """
    Instance-aware contrastive loss for hairstyle representation learning.
    
    Computes contrastive loss between teacher and student projections.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Initialize contrastive loss.
        
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        student_proj: torch.Tensor,
        teacher_proj: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.
        
        Args:
            student_proj: Student projections [B, D]
            teacher_proj: Teacher projections [B, D]
            
        Returns:
            Contrastive loss scalar
        """
        batch_size = student_proj.size(0)
        
        # Normalize projections
        student_proj = F.normalize(student_proj, dim=-1)
        teacher_proj = F.normalize(teacher_proj, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(student_proj, teacher_proj.t()) / self.temperature
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=logits.device)
        
        # Compute cross-entropy loss
        loss = F.cross_entropy(logits, labels)
        
        return loss


class HairstyleRetrievalModel(nn.Module):
    """
    Model for hairstyle-based image retrieval.
    
    Uses trained dual-view model for feature extraction and similarity computation.
    """
    
    def __init__(self, dual_view_model: DualViewHairModel):
        """
        Initialize retrieval model.
        
        Args:
            dual_view_model: Trained dual-view model
        """
        super().__init__()
        self.encoder = dual_view_model.student_encoder
        
        # Freeze parameters for inference
        for param in self.encoder.parameters():
            param.requires_grad = False
    
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """
        Encode images for retrieval.
        
        Args:
            images: Input images [B, C, H, W]
            
        Returns:
            Normalized embeddings [B, embedding_dim]
        """
        with torch.no_grad():
            embeddings = self.encoder(images, return_embedding=True)
            embeddings = F.normalize(embeddings, dim=-1)
        
        return embeddings
    
    def compute_similarity(
        self,
        query_embeddings: torch.Tensor,
        gallery_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute similarity scores between query and gallery embeddings.
        
        Args:
            query_embeddings: Query embeddings [N_q, D]
            gallery_embeddings: Gallery embeddings [N_g, D]
            
        Returns:
            Similarity matrix [N_q, N_g]
        """
        # Cosine similarity
        similarity = torch.matmul(query_embeddings, gallery_embeddings.t())
        
        return similarity
