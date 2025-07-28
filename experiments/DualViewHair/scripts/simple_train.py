"""Minimal training script for DualViewHair."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.models.dual_view_model import DualViewHairModel
from src.data.simple_dataloader import create_dataloader


class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, student_proj, teacher_proj):
        # Normalize projections
        student_proj = nn.functional.normalize(student_proj, dim=-1)
        teacher_proj = nn.functional.normalize(teacher_proj, dim=-1)
        
        # Compute similarity matrix
        logits = torch.matmul(student_proj, teacher_proj.t()) / self.temperature
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(logits.size(0), device=logits.device)
        
        # Compute cross-entropy loss
        loss = nn.functional.cross_entropy(logits, labels)
        return loss


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for batch in tqdm(loader):
        view_a = batch['view_a'].to(device)
        view_b = batch['view_b'].to(device)
        optimizer.zero_grad()
        
        # Get model outputs
        outputs = model(view_a, view_b)
        teacher_proj = outputs['teacher_projection']
        student_proj = outputs['student_projection']
        
        # Compute loss
        loss = criterion(student_proj, teacher_proj)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)


# Paths
full_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/celeb10k"
hair_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/hair_representation/hair_training_data/train/dummy"

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader = create_dataloader(full_dir, hair_dir, batch_size=80)

# Model
model = DualViewHairModel(embedding_dim=256).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = ContrastiveLoss()

# Train - no validation
for epoch in range(50):
    train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
    print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}")
    
    # Save model every 10 epochs
    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), f'model_epoch_{epoch+1}.pth')

print("Training complete!")
