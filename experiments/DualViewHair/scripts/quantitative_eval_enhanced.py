"""Quantitative evaluation for Enhanced DualViewHair model using FAISS and benchmark."""

import os
import json
import torch
import faiss
import pickle
import numpy as np
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch import nn
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models.enhanced_model import EnhancedDualViewHairModel
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# ----------------------------
# Dataset for FAISS indexing
# ----------------------------
class HairImageDataset(Dataset):
    """Dataset for loading hair images with paths."""
    
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        
        # Find all hair images
        self.image_paths = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            self.image_paths.extend(list(self.root_dir.glob(f"**/{ext}")))
        
        print(f"Found {len(self.image_paths)} images in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, str(img_path)


# ----------------------------
# Enhanced Model Wrapper
# ----------------------------
class EnhancedDualViewHairWrapper(nn.Module):
    """Wrapper for Enhanced DualViewHair model to extract features."""
    
    def __init__(self, model_path, embedding_dim=256, encoder_type='multiscale', use_cross_alignment=True, device='cuda'):
        super().__init__()
        self.device = device
        self.encoder_type = encoder_type
        self.use_cross_alignment = use_cross_alignment
        
        # Load trained enhanced model
        self.model = EnhancedDualViewHairModel(
            embedding_dim=embedding_dim,
            encoder_type=encoder_type,
            use_cross_alignment=use_cross_alignment
        )
        
        # Load checkpoint
        if model_path.endswith('.pth'):
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                # Print config info if available
                if 'config' in checkpoint:
                    config = checkpoint['config']
                    print(f"📋 Loaded config: {config['name']}")
                    print(f"   Encoder type: {config['encoder_type']}")
                    print(f"   Cross-alignment: {config.get('use_cross_alignment', 'Unknown')}")
            else:
                self.model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unsupported checkpoint format: {model_path}")
        
        self.model.to(device)
        self.model.eval()
        
        print(f"✅ Loaded Enhanced DualViewHair model from: {model_path}")
        print(f"   Architecture: {encoder_type} + cross_alignment={use_cross_alignment}")
    
    def extract_features(self, x):
        """Extract normalized embeddings for retrieval."""
        with torch.no_grad():
            # Use student encoder for feature extraction
            embeddings = self.model.student_encoder(x, return_embedding=True)
            # Normalize for cosine similarity
            embeddings = torch.nn.functional.normalize(embeddings, dim=-1)
        return embeddings


# ----------------------------
# Utilities
# ----------------------------
def get_transform():
    """Get image preprocessing transform."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_image(image_path):
    """Load and preprocess single image."""
    image = Image.open(image_path).convert("RGB")
    return get_transform()(image).unsqueeze(0)


@torch.no_grad()
def extract_feature(model, image_path, device):
    """Extract feature for single image."""
    image = load_image(image_path).to(device)
    feat = model.extract_features(image).cpu().numpy()
    faiss.normalize_L2(feat)  # Ensure L2 normalization for FAISS
    return feat


@torch.no_grad()
def build_faiss_index(model, db_path, index_dir, device):
    """Build FAISS index from database images."""
    print(f"🔍 Building FAISS index from {db_path} ...")
    os.makedirs(index_dir, exist_ok=True)

    # Create dataset
    dataset = HairImageDataset(db_path, transform=get_transform())
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)

    model.eval()
    features = []
    paths = []

    # Extract features for all database images
    for images, img_paths in tqdm(dataloader, desc="Extracting DB features"):
        images = images.to(device)
        feats = model.extract_features(images).cpu().numpy()
        features.append(feats)
        paths.extend(img_paths)

    # Combine all features
    features_np = np.vstack(features)
    faiss.normalize_L2(features_np)

    # Create FAISS index
    index = faiss.IndexFlatL2(features_np.shape[1])
    index.add(features_np)

    # Save index and paths
    faiss.write_index(index, os.path.join(index_dir, "enhanced_dualviewhair_faiss.index"))
    with open(os.path.join(index_dir, "enhanced_dualviewhair_paths.pkl"), "wb") as f:
        pickle.dump(paths, f)

    print("✅ Enhanced FAISS index created and saved.")
    return index, paths


# ----------------------------
# Main Evaluation
# ----------------------------
def evaluate(model, benchmark, index, all_paths, device, query_root, database_root):
    """Evaluate enhanced model on benchmark."""
    Ks = [10, 20, 50]
    recall_at_k = defaultdict(int)
    ap_at_k = defaultdict(list)
    total_queries = 0
    retrieval_results = []

    for item in tqdm(benchmark, desc="Evaluating Enhanced Model"):
        # Get query and ground truth paths
        query_image = query_root + (item["query_image"][:-4] + "_hair.png").split("/")[-1]
        gt_list = [query_root + (x[:-4] + "_hair.png").split("/")[-1] for x in item["ground_truth"]]
        query_path = os.path.join(database_root, query_image)

        # Check if files exist
        if not os.path.exists(query_path):
            print(f"[WARN] Missing query: {query_path}")
            continue
        if not all(os.path.exists(os.path.join(database_root, gt)) for gt in gt_list):
            print(f"[WARN] Missing GT for: {query_image}")
            continue

        # Extract query feature and search
        query_feat = extract_feature(model, query_path, device)
        scores, indices = index.search(query_feat, max(Ks))
        retrieved = [all_paths[i] for i in indices[0]]

        # Store results
        retrieval_results.append({
            "query": query_path.split("/")[-1],
            "top100": [x.split("/")[-1] for x in retrieved[:100]]
        })

        # Compute metrics for each K
        for k in Ks:
            top_k_preds = retrieved[:k]
            
            # Recall@K: check if any ground truth is in top-k
            if any(gt in top_k_preds for gt in gt_list):
                recall_at_k[k] += 1
                
            # Average Precision@K
            hits, sum_precisions = 0, 0
            for i, p in enumerate(top_k_preds):
                if p in gt_list:
                    hits += 1
                    sum_precisions += hits / (i + 1)
            ap = sum_precisions / min(len(gt_list), k) if gt_list else 0.0
            ap_at_k[k].append(ap)

        total_queries += 1

    # Save retrieval results
    output_json_path = os.path.join("log_json", "enhanced_dualviewhair_top100_results.json")
    os.makedirs("log_json", exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(retrieval_results, f, indent=2)
    print(f"\\n📁 Top-100 retrieval results saved to {output_json_path}")

    # Print evaluation results
    print("\\n📈 Enhanced DualViewHair Evaluation Results:")
    print("=" * 60)
    for k in Ks:
        mean_ap = sum(ap_at_k[k]) / len(ap_at_k[k]) if ap_at_k[k] else 0
        recall = recall_at_k[k] / total_queries if total_queries > 0 else 0
        print(f"mAP@{k:2d}: {mean_ap:.4f}")
        print(f"R@{k:2d}:   {recall:.4f}")
        print("-" * 25)
    
    return {
        'mAP': {k: sum(ap_at_k[k]) / len(ap_at_k[k]) if ap_at_k[k] else 0 for k in Ks},
        'Recall': {k: recall_at_k[k] / total_queries if total_queries > 0 else 0 for k in Ks},
        'total_queries': total_queries
    }


# ----------------------------
# Model Detection
# ----------------------------
def detect_model_config(checkpoint_path):
    """Detect model configuration from checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'config' in checkpoint:
            config = checkpoint['config']
            return config['encoder_type'], config.get('use_cross_alignment', True)
        else:
            # Try to infer from filename
            filename = os.path.basename(checkpoint_path).lower()
            if 'multiscale' in filename:
                return 'multiscale', True
            elif 'partbased' in filename:
                return 'partbased', True
            elif 'standard' in filename:
                return 'standard', True
            else:
                print("⚠️  Could not detect model config, using default: multiscale + cross_alignment")
                return 'multiscale', True
    except Exception as e:
        print(f"⚠️  Error detecting config: {e}, using default")
        return 'multiscale', True


# ----------------------------
# Main Entry
# ----------------------------
def main():
    """Main evaluation function for enhanced model."""
    
    # === CONFIG ===
    benchmark_path = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/hairstyle_retrieval_benchmark.json"
    database_root = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/HairLearning/data/train"
    query_root = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/HairLearning/data/train/HairImages/"
    
    # Model checkpoint - adjust this path
    checkpoint_path = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/DualViewHair/enhanced_model_MultiScale_Attention_epoch_20.pth"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"🚀 Using enhanced checkpoint: {checkpoint_path}")

    # Detect model configuration
    encoder_type, use_cross_alignment = detect_model_config(checkpoint_path)
    
    # Create index directory based on model config
    index_dir = f"./faiss_index_enhanced_{encoder_type}{'_aligned' if use_cross_alignment else ''}"
    
    # Load benchmark
    if not os.path.exists(benchmark_path):
        print(f"❌ Benchmark file not found: {benchmark_path}")
        return
    
    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)
    print(f"📋 Loaded benchmark with {len(benchmark)} queries")

    # Load enhanced model
    try:
        model = EnhancedDualViewHairWrapper(
            checkpoint_path, 
            embedding_dim=256, 
            encoder_type=encoder_type,
            use_cross_alignment=use_cross_alignment,
            device=device
        )
    except Exception as e:
        print(f"❌ Failed to load enhanced model: {e}")
        return

    # Load or create FAISS index
    index_path = os.path.join(index_dir, "enhanced_dualviewhair_faiss.index")
    paths_path = os.path.join(index_dir, "enhanced_dualviewhair_paths.pkl")

    if os.path.exists(index_path) and os.path.exists(paths_path):
        print("📦 Loading existing enhanced FAISS index...")
        index = faiss.read_index(index_path)
        with open(paths_path, "rb") as f:
            all_paths = pickle.load(f)
        print(f"✅ Loaded enhanced index with {index.ntotal} images")
    else:
        print("🔨 Building new enhanced FAISS index...")
        index, all_paths = build_faiss_index(model, database_root, index_dir, device)

    # Run evaluation
    results = evaluate(model, benchmark, index, all_paths, device, query_root, database_root)
    
    # Save results summary
    results_summary = {
        'model': checkpoint_path,
        'model_type': 'enhanced_dualviewhair',
        'model_config': {
            'encoder_type': encoder_type,
            'use_cross_alignment': use_cross_alignment,
            'embedding_dim': 256
        },
        'benchmark': benchmark_path,
        'metrics': results,
        'config': {
            'device': device,
            'index_size': index.ntotal,
            'index_dir': index_dir
        }
    }
    
    # Create filename with model config
    summary_filename = f"enhanced_dualviewhair_{encoder_type}{'_aligned' if use_cross_alignment else ''}_evaluation_summary.json"
    
    with open(summary_filename, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\\n💾 Enhanced evaluation summary saved to: {summary_filename}")
    
    # Compare with baseline if available
    baseline_summary = "dualviewhair_evaluation_summary.json"
    if os.path.exists(baseline_summary):
        print("\\n📊 Performance Comparison:")
        print("=" * 60)
        
        with open(baseline_summary, "r") as f:
            baseline_results = json.load(f)
        
        baseline_metrics = baseline_results['metrics']
        enhanced_metrics = results
        
        for metric_type in ['mAP', 'Recall']:
            print(f"\\n{metric_type}:")
            for k in [10, 20, 50]:
                baseline_val = baseline_metrics[metric_type].get(str(k), 0)
                enhanced_val = enhanced_metrics[metric_type].get(k, 0)
                improvement = ((enhanced_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
                
                print(f"  @{k:2d}: Baseline={baseline_val:.4f} → Enhanced={enhanced_val:.4f} "
                      f"({improvement:+.1f}%)")


if __name__ == '__main__':
    main()
