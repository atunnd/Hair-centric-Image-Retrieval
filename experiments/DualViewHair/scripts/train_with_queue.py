"""Enhanced training with momentum queue for larger negative pools."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import torch.nn.functional as F

from src.models.dual_view_model import DualViewHairModel
from src.data.simple_dataloader import create_dataloader


class MomentumQueue:
    """Momentum queue for maintaining large pool of negative embeddings."""
    
    def __init__(self, dim: int, queue_size: int = 4096):
        self.dim = dim
        self.queue_size = queue_size
        self.ptr = 0
        
        # Initialize queue with random embeddings
        self.register_buffer = lambda name, tensor: setattr(self, name, tensor)
        self.register_buffer('queue', torch.randn(dim, queue_size))
        self.queue = F.normalize(self.queue, dim=0)
    
    def to(self, device):
        self.queue = self.queue.to(device)
        return self
    
    @torch.no_grad()
    def dequeue_and_enqueue(self, keys):
        """Update queue with new embeddings."""
        batch_size = keys.shape[0]
        
        # Replace oldest embeddings
        if self.ptr + batch_size <= self.queue_size:
            self.queue[:, self.ptr:self.ptr + batch_size] = keys.T
        else:
            # Handle wraparound
            remaining = self.queue_size - self.ptr
            self.queue[:, self.ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
        
        self.ptr = (self.ptr + batch_size) % self.queue_size
    
    def get_queue(self):
        """Get current queue of negative embeddings."""
        return self.queue.clone()


class ContrastiveLossWithQueue(nn.Module):
    """Contrastive loss with momentum queue for large negative pools."""
    
    def __init__(self, temperature=0.07, queue_size=4096):
        super().__init__()
        self.temperature = temperature
        self.queue_size = queue_size
        self.teacher_queue = None
    
    def init_queues(self, embed_dim, device):
        """Initialize momentum queue for teacher embeddings."""
        self.teacher_queue = MomentumQueue(embed_dim, self.queue_size).to(device)
    
    def forward(self, student_proj, teacher_proj):
        batch_size = student_proj.size(0)
        
        # Normalize projections
        student_proj = F.normalize(student_proj, dim=-1)
        teacher_proj = F.normalize(teacher_proj, dim=-1)
        
        # Initialize queues if needed
        if self.teacher_queue is None:
            self.init_queues(student_proj.size(1), student_proj.device)
        
        # Get current queues
        teacher_queue = self.teacher_queue.get_queue()  # [embed_dim, queue_size]
        
        # Compute similarities
        # Positive pairs: student with corresponding teacher (diagonal)
        pos_sim = torch.sum(student_proj * teacher_proj, dim=1, keepdim=True)  # [B, 1]
        
        # Negative pairs: student with teacher queue ONLY
        # Don't include current batch in negatives to avoid confusion
        neg_sim = torch.mm(student_proj, teacher_queue)  # [B, queue_size]
        
        # Combine positive and negative similarities
        logits = torch.cat([pos_sim, neg_sim], dim=1) / self.temperature  # [B, 1 + queue_size]
        
        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)
        
        # Compute loss
        loss = F.cross_entropy(logits, labels)
        
        # Update queue with current teacher embeddings
        with torch.no_grad():
            self.teacher_queue.dequeue_and_enqueue(teacher_proj)
        
        return loss


def train_epoch_with_queue(model, loader, optimizer, criterion, device):
    """Training epoch with momentum queue."""
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        view_a = batch['view_a'].to(device)
        view_b = batch['view_b'].to(device)
        optimizer.zero_grad()
        
        # Get model outputs
        outputs = model(view_a, view_b)
        teacher_proj = outputs['teacher_projection']
        student_proj = outputs['student_projection']
        
        # Compute loss with queue
        loss = criterion(student_proj, teacher_proj)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)


def main_with_queue():
    """Main training with momentum queue."""
    
    # Paths
    full_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/celeb10k"
    hair_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/hair_representation/hair_training_data/train/dummy"
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_loader = create_dataloader(full_dir, hair_dir, batch_size=80)  # Can use larger batch
    
    # Model
    model = DualViewHairModel(embedding_dim=256, projection_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Enhanced criterion with momentum queue
    criterion = ContrastiveLossWithQueue(temperature=0.07, queue_size=2048)
    
    print(f"Training with momentum queue (size: {criterion.queue_size})")
    print(f"Device: {device}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Dataset size: {len(train_loader.dataset)}")
    
    # Train
    for epoch in range(50):
        train_loss = train_epoch_with_queue(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}")
        
        # Save model every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': train_loss,
            }, f'model_checkpoints/model_with_queue_epoch_{epoch+1}.pth')
            print(f"  Saved: model_with_queue_epoch_{epoch+1}.pth")
    
    print("Training with momentum queue complete!")


if __name__ == "__main__":
    main_with_queue()
