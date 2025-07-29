"""
Clean loss function implementations for DualViewHair.

All contrastive loss variants with consistent interfaces.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class BaseLoss(nn.Module):
    """Base class for all loss functions."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute loss from model outputs.
        
        Args:
            outputs: Dictionary from DualViewHairModel.forward()
            
        Returns:
            Dictionary with loss components
        """
        raise NotImplementedError


class InfoNCELoss(BaseLoss):
    """InfoNCE contrastive loss (original implementation)."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__(temperature)
    
    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        student_proj = outputs['student_projection']  # [B, D]
        teacher_proj = outputs['teacher_projection']  # [B, D]
        
        # Normalize projections
        student_proj = F.normalize(student_proj, dim=1)
        teacher_proj = F.normalize(teacher_proj, dim=1)
        
        # Compute similarity matrix
        similarity = torch.matmul(student_proj, teacher_proj.T) / self.temperature  # [B, B]
        
        # Positive pairs are on the diagonal
        labels = torch.arange(similarity.size(0)).to(similarity.device)
        
        # InfoNCE loss
        loss = F.cross_entropy(similarity, labels)
        
        return {
            'loss': loss,
            'contrastive_loss': loss
        }


class NTXentLoss(BaseLoss):
    """NT-Xent contrastive loss (bidirectional)."""
    
    def __init__(self, temperature: float = 0.1):
        super().__init__(temperature)
    
    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        student_proj = outputs['student_projection']  # [B, D]
        teacher_proj = outputs['teacher_projection']  # [B, D]
        
        # Normalize projections
        student_proj = F.normalize(student_proj, dim=1)
        teacher_proj = F.normalize(teacher_proj, dim=1)
        
        batch_size = student_proj.size(0)
        device = student_proj.device
        
        # Create augmented batch by concatenating student and teacher
        representations = torch.cat([student_proj, teacher_proj], dim=0)  # [2B, D]
        
        # Compute full similarity matrix
        similarity = torch.matmul(representations, representations.T) / self.temperature  # [2B, 2B]
        
        # Create labels for positive pairs
        # For NT-Xent: positive pairs are (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size),  # Student -> Teacher
            torch.arange(batch_size)                   # Teacher -> Student
        ]).to(device)
        
        # Remove self-similarities (diagonal)
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(device)
        similarity = similarity.masked_fill(mask, -float('inf'))
        
        # Compute NT-Xent loss
        loss = F.cross_entropy(similarity, labels)
        
        return {
            'loss': loss,
            'contrastive_loss': loss
        }


class AsymmetricNTXentLoss(BaseLoss):
    """Asymmetric NT-Xent loss with different weights for directions."""
    
    def __init__(self, temperature: float = 0.1, weight_s2t: float = 1.0, weight_t2s: float = 0.5):
        super().__init__(temperature)
        self.weight_s2t = weight_s2t  # Student to Teacher weight
        self.weight_t2s = weight_t2s  # Teacher to Student weight
    
    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        student_proj = outputs['student_projection']  # [B, D]
        teacher_proj = outputs['teacher_projection']  # [B, D]
        
        # Normalize projections
        student_proj = F.normalize(student_proj, dim=1)
        teacher_proj = F.normalize(teacher_proj, dim=1)
        
        batch_size = student_proj.size(0)
        device = student_proj.device
        
        # Student to Teacher loss
        s2t_similarity = torch.matmul(student_proj, teacher_proj.T) / self.temperature
        s2t_labels = torch.arange(batch_size).to(device)
        s2t_loss = F.cross_entropy(s2t_similarity, s2t_labels)
        
        # Teacher to Student loss
        t2s_similarity = torch.matmul(teacher_proj, student_proj.T) / self.temperature
        t2s_labels = torch.arange(batch_size).to(device)
        t2s_loss = F.cross_entropy(t2s_similarity, t2s_labels)
        
        # Weighted combination
        total_loss = self.weight_s2t * s2t_loss + self.weight_t2s * t2s_loss
        
        return {
            'loss': total_loss,
            'contrastive_loss': total_loss,
            's2t_loss': s2t_loss,
            't2s_loss': t2s_loss
        }


class AlignmentLoss(BaseLoss):
    """Cross-view alignment loss for enhanced models."""
    
    def __init__(self, temperature: float = 0.1, alignment_weight: float = 0.1):
        super().__init__(temperature)
        self.alignment_weight = alignment_weight
    
    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        if 'aligned_student_embedding' not in outputs:
            raise ValueError("Alignment loss requires aligned embeddings from model")
        
        # Original embeddings
        student_emb = outputs['student_embedding']
        teacher_emb = outputs['teacher_embedding']
        
        # Aligned embeddings
        aligned_student_emb = outputs['aligned_student_embedding']
        aligned_teacher_emb = outputs['aligned_teacher_embedding']
        
        # Alignment loss: encourage alignment while preserving original representations
        student_alignment_loss = F.mse_loss(aligned_student_emb, student_emb)
        teacher_alignment_loss = F.mse_loss(aligned_teacher_emb, teacher_emb)
        
        alignment_loss = (student_alignment_loss + teacher_alignment_loss) / 2
        
        return {
            'alignment_loss': alignment_loss * self.alignment_weight
        }


class HybridLoss(BaseLoss):
    """Hybrid loss combining contrastive and alignment losses."""
    
    def __init__(
        self,
        contrastive_loss: str = "ntxent",
        temperature: float = 0.1,
        alignment_weight: float = 0.1,
        **contrastive_kwargs
    ):
        super().__init__(temperature)
        
        # Create contrastive loss
        loss_classes = {
            'infonce': InfoNCELoss,
            'ntxent': NTXentLoss,
            'asymmetric_ntxent': AsymmetricNTXentLoss
        }
        
        if contrastive_loss not in loss_classes:
            raise ValueError(f"Unknown contrastive loss: {contrastive_loss}")
        
        self.contrastive_criterion = loss_classes[contrastive_loss](
            temperature=temperature, **contrastive_kwargs
        )
        
        # Create alignment loss
        self.alignment_criterion = AlignmentLoss(
            temperature=temperature,
            alignment_weight=alignment_weight
        )
    
    def forward(self, outputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Compute contrastive loss
        contrastive_outputs = self.contrastive_criterion(outputs)
        
        # Compute alignment loss if aligned embeddings are available
        if 'aligned_student_embedding' in outputs:
            alignment_outputs = self.alignment_criterion(outputs)
            
            # Combine losses
            total_loss = contrastive_outputs['loss'] + alignment_outputs['alignment_loss']
            
            result = {
                'loss': total_loss,
                **contrastive_outputs,
                **alignment_outputs
            }
        else:
            result = contrastive_outputs
        
        return result


def create_loss_function(loss_config) -> BaseLoss:
    """Factory function to create loss functions from config."""
    loss_type = loss_config.loss_type
    
    if loss_type == "infonce":
        return InfoNCELoss(temperature=loss_config.temperature)
    
    elif loss_type == "ntxent":
        return NTXentLoss(temperature=loss_config.temperature)
    
    elif loss_type == "asymmetric_ntxent":
        return AsymmetricNTXentLoss(
            temperature=loss_config.temperature,
            weight_s2t=loss_config.weight_s2t,
            weight_t2s=loss_config.weight_t2s
        )
    
    elif loss_type == "hybrid":
        return HybridLoss(
            contrastive_loss=loss_config.contrastive_loss,
            temperature=loss_config.temperature,
            alignment_weight=loss_config.alignment_weight
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
