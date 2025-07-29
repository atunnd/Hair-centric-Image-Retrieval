"""
Enhanced DualViewHair model incorporating fine-grained detail capture techniques.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from src.models.enhanced_dual_view import MultiScaleHairEncoder, PartBasedHairEncoder, CrossViewAlignment


class EnhancedDualViewHairModel(nn.Module):
    """
    Enhanced dual-view model with multiple fine-grained detail capture mechanisms.
    
    Features:
    1. Multi-scale feature extraction
    2. Spatial attention for hair focus
    3. Part-based hair learning
    4. Cross-view feature alignment
    """
    
    def __init__(
        self,
        backbone: str = 'resnet50',
        pretrained: bool = True,
        embedding_dim: int = 256,
        projection_dim: int = 128,
        momentum: float = 0.999,
        encoder_type: str = 'multiscale',  # 'multiscale', 'partbased', or 'standard'
        use_cross_alignment: bool = True
    ):
        """
        Initialize enhanced dual-view model.
        
        Args:
            encoder_type: Type of encoder ('multiscale', 'partbased', 'standard')
            use_cross_alignment: Whether to use cross-view alignment
        """
        super().__init__()
        
        self.momentum = momentum
        self.encoder_type = encoder_type
        self.use_cross_alignment = use_cross_alignment
        
        # Choose encoder architecture
        if encoder_type == 'multiscale':
            encoder_class = MultiScaleHairEncoder
            encoder_kwargs = {'use_attention': True}
        elif encoder_type == 'partbased':
            encoder_class = PartBasedHairEncoder
            encoder_kwargs = {'num_parts': 3}
        else:
            # Fall back to standard encoder
            from src.models.dual_view_model import HairstyleEncoder
            encoder_class = HairstyleEncoder
            encoder_kwargs = {}
        
        # Student encoder (processes full images)
        self.student_encoder = encoder_class(
            backbone=backbone,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            **encoder_kwargs
        )
        
        # Teacher encoder (processes hair-only images)
        self.teacher_encoder = encoder_class(
            backbone=backbone,
            pretrained=pretrained,
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            **encoder_kwargs
        )
        
        # Cross-view alignment module
        if self.use_cross_alignment:
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
        view_a: torch.Tensor,
        view_b: torch.Tensor,
        update_teacher: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Enhanced forward pass with fine-grained detail capture.
        """
        
        # Student forward pass (full images)
        student_proj = self.student_encoder(view_b, return_embedding=False)
        student_emb = self.student_encoder(view_b, return_embedding=True)
        
        # Teacher forward pass (hair-only images)
        with torch.no_grad():
            teacher_proj = self.teacher_encoder(view_a, return_embedding=False)
            teacher_emb = self.teacher_encoder(view_a, return_embedding=True)
        
        # Cross-view alignment for better feature correspondence
        if self.use_cross_alignment:
            aligned_teacher_emb, aligned_student_emb = self.cross_alignment(
                teacher_emb, student_emb
            )
            
            # Recompute projections with aligned embeddings
            aligned_teacher_proj = self.teacher_encoder.projection_head(aligned_teacher_emb)
            aligned_student_proj = self.student_encoder.projection_head(aligned_student_emb)
        else:
            aligned_teacher_proj = teacher_proj
            aligned_student_proj = student_proj
            aligned_teacher_emb = teacher_emb
            aligned_student_emb = student_emb
        
        # Update teacher parameters
        if update_teacher and self.training:
            self._update_teacher()
        
        return {
            'student_projection': aligned_student_proj,
            'student_embedding': aligned_student_emb,
            'teacher_projection': aligned_teacher_proj,
            'teacher_embedding': aligned_teacher_emb,
            # Also return original features for analysis
            'original_student_projection': student_proj,
            'original_teacher_projection': teacher_proj,
            'original_student_embedding': student_emb,
            'original_teacher_embedding': teacher_emb
        }
    
    def get_embeddings(self, images: torch.Tensor, use_teacher: bool = False) -> torch.Tensor:
        """Extract embeddings for retrieval."""
        encoder = self.teacher_encoder if use_teacher else self.student_encoder
        
        with torch.no_grad():
            embeddings = encoder(images, return_embedding=True)
        
        return embeddings


class HybridLoss(nn.Module):
    """
    Hybrid loss combining multiple objectives for fine-grained learning.
    """
    
    def __init__(
        self,
        temperature: float = 0.07,
        alignment_weight: float = 0.1,
        consistency_weight: float = 0.05
    ):
        super().__init__()
        self.temperature = temperature
        self.alignment_weight = alignment_weight
        self.consistency_weight = consistency_weight
    
    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute hybrid loss with multiple objectives.
        
        Returns:
            Dictionary with individual loss components
        """
        
        # Main contrastive loss (aligned features)
        student_proj = outputs['student_projection']
        teacher_proj = outputs['teacher_projection']
        
        batch_size = student_proj.size(0)
        
        # Normalize projections
        student_proj = F.normalize(student_proj, dim=-1)
        teacher_proj = F.normalize(teacher_proj, dim=-1)
        
        # Contrastive loss
        logits = torch.matmul(student_proj, teacher_proj.t()) / self.temperature
        labels = torch.arange(batch_size, device=logits.device)
        contrastive_loss = F.cross_entropy(logits, labels)
        
        # Alignment consistency loss (if cross-alignment is used)
        alignment_loss = torch.tensor(0.0, device=student_proj.device)
        if 'original_student_projection' in outputs:
            orig_student = F.normalize(outputs['original_student_projection'], dim=-1)
            orig_teacher = F.normalize(outputs['original_teacher_projection'], dim=-1)
            
            # Encourage aligned features to maintain similarity to originals
            student_consistency = F.mse_loss(student_proj, orig_student)
            teacher_consistency = F.mse_loss(teacher_proj, orig_teacher)
            alignment_loss = (student_consistency + teacher_consistency) / 2
        
        # Total loss
        total_loss = (
            contrastive_loss + 
            self.alignment_weight * alignment_loss
        )
        
        return {
            'total_loss': total_loss,
            'contrastive_loss': contrastive_loss,
            'alignment_loss': alignment_loss
        }
