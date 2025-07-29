"""
Simple training script for enhanced DualViewHair with fine-grained details.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

import torch
import torch.optim as optim
from tqdm import tqdm

from src.data.simple_dataloader import create_dataloader
from src.models.enhanced_model import EnhancedDualViewHairModel, HybridLoss
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


def train_epoch_enhanced(model, loader, optimizer, criterion, device):
    """Training epoch with enhanced model and hybrid loss."""
    model.train()
    total_losses = {'total': 0, 'contrastive': 0, 'alignment': 0}
    
    for batch in tqdm(loader, desc="Enhanced Training"):
        view_a = batch['view_a'].to(device)  # Hair regions
        view_b = batch['view_b'].to(device)  # Full images
        
        optimizer.zero_grad()
        
        # Forward pass
        outputs = model(view_a, view_b)
        
        # Compute hybrid loss
        loss_dict = criterion(outputs)
        total_loss = loss_dict['total_loss']
        
        # Backward pass
        total_loss.backward()
        optimizer.step()
        
        # Accumulate losses
        for key in total_losses:
            if key == 'total':
                total_losses[key] += total_loss.item()
            else:
                total_losses[key] += loss_dict[f'{key}_loss'].item()
    
    # Average losses
    for key in total_losses:
        total_losses[key] /= len(loader)
    
    return total_losses


def main():
    """Main training function with enhanced architecture options."""
    
    # Paths
    full_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/celeb10k"
    hair_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/hair_representation/hair_training_data/train/dummy"
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Data
    train_loader = create_dataloader(
        full_dir, hair_dir,
        batch_size=100,
        num_workers=4
    )
    
    # Model configurations to try
    model_configs = [
        {
            'name': 'MultiScale_Attention',
            'encoder_type': 'multiscale',
            'use_cross_alignment': True
        },
        {
            'name': 'PartBased_3Parts',
            'encoder_type': 'partbased',
            'use_cross_alignment': True
        },
        {
            'name': 'Standard_CrossAlign',
            'encoder_type': 'standard',
            'use_cross_alignment': True
        }
    ]
    
    # Choose configuration (or loop through all)
    config = model_configs[0]  # Start with multi-scale
    
    print(f"Training Enhanced DualViewHair: {config['name']}")
    print(f"Device: {device}")
    print(f"Batch size: {train_loader.batch_size}")
    print(f"Dataset size: {len(train_loader.dataset)}")
    
    # Model
    model = EnhancedDualViewHairModel(
        embedding_dim=256,
        projection_dim=128,
        encoder_type=config['encoder_type'],
        use_cross_alignment=config['use_cross_alignment']
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.0008)  # Slightly lower LR
    
    # Enhanced loss function
    criterion = HybridLoss(
        temperature=0.07,
        alignment_weight=0.1,
        consistency_weight=0.05
    )
    
    # Training loop
    for epoch in range(40):  # Fewer epochs due to enhanced learning
        losses = train_epoch_enhanced(model, train_loader, optimizer, criterion, device)
        
        print(f"Epoch {epoch+1:2d}: "
              f"Total={losses['total']:.4f}, "
              f"Contrastive={losses['contrastive']:.4f}, "
              f"Alignment={losses['alignment']:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"enhanced_model_{config['name']}_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'losses': losses,
                'config': config
            }, checkpoint_path)
            print(f"  Saved: {checkpoint_path}")
    
    print(f"Enhanced training complete for {config['name']}!")


def compare_architectures():
    """Quick comparison of different enhanced architectures."""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4
    
    # Dummy input
    view_a = torch.randn(batch_size, 3, 224, 224).to(device)
    view_b = torch.randn(batch_size, 3, 224, 224).to(device)
    
    configs = [
        ('MultiScale', 'multiscale', True),
        ('PartBased', 'partbased', True),
        ('Standard+Align', 'standard', True),
        ('Standard', 'standard', False)
    ]
    
    print("Architecture Comparison:")
    print("-" * 50)
    
    for name, encoder_type, use_alignment in configs:
        model = EnhancedDualViewHairModel(
            encoder_type=encoder_type,
            use_cross_alignment=use_alignment
        ).to(device)
        
        # Forward pass
        with torch.no_grad():
            outputs = model(view_a, view_b)
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"{name:15s}: {trainable_params/1e6:.1f}M params, "
              f"Output shape: {outputs['student_projection'].shape}")


if __name__ == "__main__":
    print("Enhanced DualViewHair Training")
    print("=" * 50)
    
    # Quick architecture comparison
    # compare_architectures()
    
    # Main training
    main()
