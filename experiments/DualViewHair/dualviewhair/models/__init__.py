"""
DualViewHair models package.
"""

from .models import (
    DualViewHairModel, 
    BaselineEncoder, 
    MultiScaleEncoder, 
    PartBasedEncoder
)
from .components import (
    BaseEncoder,
    ResNetBackbone,
    EmbeddingHead,
    ProjectionHead,
    SpatialAttention,
    CrossViewAlignment
)

__all__ = [
    "DualViewHairModel",
    "BaselineEncoder", 
    "MultiScaleEncoder",
    "PartBasedEncoder",
    "BaseEncoder",
    "ResNetBackbone",
    "EmbeddingHead", 
    "ProjectionHead",
    "SpatialAttention",
    "CrossViewAlignment"
]
