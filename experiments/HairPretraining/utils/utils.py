import torch
import random
import numpy as np

import torch
import torch.nn.functional as F

def mse_alignment_loss(model, positives, masked_positives, beta=0.05):
    """
    Tính MSE loss giữa embeddings của positive và masked positive,
    với detach gradient từ masked branch.
    
    Args:
    - model: nn.Module, backbone + projection (trả về normalized embeddings)
    - positives: torch.Tensor (B, C, H, W), batch positive images
    - masked_positives: torch.Tensor (B, C, H, W), batch masked positives
    - beta: float, weight cho MSE loss
    
    Returns:
    - mse_loss: torch.Tensor, beta * MSE (scalar)
    """
    # Forward qua model
    emb_pos = model(positives)  # (B, projection_dim), normalized
    emb_masked = model(masked_positives).detach()  # Detach để cắt gradient
    
    # Tính MSE
    mse = F.mse_loss(emb_pos, emb_masked, reduction='mean')
    
    return beta * mse

def get_optimizer(model, lr, weight_decay, beta1, beta2):
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if name.endswith(".bias") or "bn" in name or "norm" in name:
                no_decay.append(param)
            else:
                decay.append(param)
    param_groups = [
        {'params': decay, 'weight_decay': weight_decay},
        {'params': no_decay, 'weight_decay': 0.0}
    ]
    return torch.optim.Adam(param_groups, lr, betas=(beta1, beta2))

def linear_increase_alpha(start_alpha, current_epoch, max_epochs, alpha_max=0.9):
    alpha = start_alpha + (alpha_max - start_alpha) * (current_epoch / max_epochs)
    return min(alpha_max, alpha)

def margin_decay(epoch: int, total_epochs: int,
                 min_margin: float = 0.1, 
                 max_margin: float = 0.9, 
                 step: float = 0.05) -> float:
    """
    Tính margin theo lịch decay tuyến tính từ max_margin -> min_margin.
    
    Args:
        epoch (int): epoch hiện tại (bắt đầu từ 0).
        total_epochs (int): tổng số epoch.
        min_margin (float): margin nhỏ nhất.
        max_margin (float): margin lớn nhất.
        step (float): bước làm tròn.
    
    Returns:
        float: margin tại epoch hiện tại (đã làm tròn theo step).
    """
    # hệ số t trong [0,1]
    t = epoch / (total_epochs - 1)
    # decay tuyến tính
    margin = max_margin - (max_margin - min_margin) * t
    # làm tròn theo step
    margin = round(margin / step) * step
    # clamp vào [min_margin, max_margin]
    margin = max(min_margin, min(max_margin, margin))
    return margin


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def update_momentum(student, teacher, m):
    for s_param, t_param in zip(student.parameters(), teacher.parameters()):
        t_param.data = m * t_param.data + (1 - m) * s_param.data