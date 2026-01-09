import torch
import random
import numpy as np
from collections import defaultdict, Counter
import os
import torchvision
import torch.nn as nn
import torch.nn.functional as F

def NegSamplerRandomly(embeddings: torch.Tensor):
    """
    Randomly shuffle embeddings to get negatives without self-pairs.
    """
    batch_size = embeddings.size(0)
    perm = torch.randperm(batch_size, device=embeddings.device)

    # Nếu có vị trí trùng chính nó thì shift đi 1
    for i in range(batch_size):
        if perm[i] == i:
            perm[i] = (perm[i] + 1) % batch_size
    
    shuffled_embeddings = embeddings[perm]
    return shuffled_embeddings


def NegSamplerStatic(model, batch, metric="cosine", k=7):
    # create embeddings
    with torch.no_grad():
        embeddings = model.extract_features_ema(batch)

    B, D = embeddings.shape
    
    if metric == 'cosine':
        # Normalize embeddings for cosine similarity
        embeddings_norm = embeddings / torch.norm(embeddings, dim=1, keepdim=True).clamp(min=1e-8)
        # Similarity matrix: (B, B)
        sim_matrix = torch.mm(embeddings_norm, embeddings_norm.t())
    elif metric == 'euclidean':
        # Use negative distances as similarity (higher is better)
        dist_matrix = torch.cdist(embeddings, embeddings)
        sim_matrix = -dist_matrix
    else:
        raise ValueError("Unsupported metric. Choose 'cosine' or 'euclidean'.")
    
    # Sort similarities descending and get indices
    sorted_sim, sorted_indices = torch.sort(sim_matrix, dim=1, descending=True)
    
    # Return the k-th index (adjust for 0-based indexing)
    if k < 1 or k > B:
        raise ValueError(f"k must be between 1 and {B}")
    kth_indices = sorted_indices[:, k-1]
    
    return kth_indices
