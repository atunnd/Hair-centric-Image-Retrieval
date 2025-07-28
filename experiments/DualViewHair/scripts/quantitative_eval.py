"""Quantitative evaluation for DualViewHair model using FAISS and benchmark."""

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

from src.models.dual_view_model import DualViewHairModel


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
# Model Wrapper
# ----------------------------
class DualViewHairWrapper(nn.Module):
    """Wrapper for DualViewHair model to extract features."""
    
    def __init__(self, model_path, embedding_dim=256, device='cuda'):
        super().__init__()
        self.device = device
        
        # Load trained model
        self.model = DualViewHairModel(embedding_dim=embedding_dim)
        
        # Load checkpoint
        if model_path.endswith('.pth'):
            # Simple state dict
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.model.load_state_dict(checkpoint)
        else:
            raise ValueError(f"Unsupported checkpoint format: {model_path}")
        
        self.model.to(device)
        self.model.eval()
        
        print(f"✅ Loaded DualViewHair model from: {model_path}")
    
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
    faiss.write_index(index, os.path.join(index_dir, "dualviewhair_faiss.index"))
    with open(os.path.join(index_dir, "dualviewhair_paths.pkl"), "wb") as f:
        pickle.dump(paths, f)

    print("✅ FAISS index created and saved.")
    return index, paths


# ----------------------------
# Main Evaluation
# ----------------------------
def evaluate(model, benchmark, index, all_paths, device, query_root, database_root):
    """Evaluate model on benchmark."""
    Ks = [10, 20, 50]
    recall_at_k = defaultdict(int)
    ap_at_k = defaultdict(list)
    total_queries = 0
    retrieval_results = []

    for item in tqdm(benchmark, desc="Evaluating"):
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
    output_json_path = os.path.join("log_json", "dualviewhair_top100_results.json")
    os.makedirs("log_json", exist_ok=True)
    with open(output_json_path, "w") as f:
        json.dump(retrieval_results, f, indent=2)
    print(f"\\n📁 Top-100 retrieval results saved to {output_json_path}")

    # Print evaluation results
    print("\\n📈 DualViewHair Evaluation Results:")
    print("=" * 50)
    for k in Ks:
        mean_ap = sum(ap_at_k[k]) / len(ap_at_k[k]) if ap_at_k[k] else 0
        recall = recall_at_k[k] / total_queries if total_queries > 0 else 0
        print(f"mAP@{k:2d}: {mean_ap:.4f}")
        print(f"R@{k:2d}:   {recall:.4f}")
        print("-" * 20)
    
    return {
        'mAP': {k: sum(ap_at_k[k]) / len(ap_at_k[k]) if ap_at_k[k] else 0 for k in Ks},
        'Recall': {k: recall_at_k[k] / total_queries if total_queries > 0 else 0 for k in Ks},
        'total_queries': total_queries
    }


# ----------------------------
# Main Entry
# ----------------------------
def main():
    """Main evaluation function."""
    
    # === CONFIG ===
    benchmark_path = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/hairstyle_retrieval_benchmark.json"
    index_dir = "./faiss_index_dualviewhair"
    database_root = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/HairLearning/data/train"
    query_root = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/HairLearning/data/train/HairImages/"
    
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    checkpoint_path = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/experiments/DualViewHair/model_checkpoints/model_epoch_40.pth"
    print(f"🚀 Using checkpoint: {checkpoint_path}")

    # Load benchmark
    if not os.path.exists(benchmark_path):
        print(f"❌ Benchmark file not found: {benchmark_path}")
        return
    
    with open(benchmark_path, "r") as f:
        benchmark = json.load(f)
    print(f"📋 Loaded benchmark with {len(benchmark)} queries")

    # Load model
    try:
        model = DualViewHairWrapper(checkpoint_path, embedding_dim=256, device=device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Load or create FAISS index
    index_path = os.path.join(index_dir, "dualviewhair_faiss.index")
    paths_path = os.path.join(index_dir, "dualviewhair_paths.pkl")

    if os.path.exists(index_path) and os.path.exists(paths_path):
        print("📦 Loading existing FAISS index...")
        index = faiss.read_index(index_path)
        with open(paths_path, "rb") as f:
            all_paths = pickle.load(f)
        print(f"✅ Loaded index with {index.ntotal} images")
    else:
        print("🔨 Building new FAISS index...")
        index, all_paths = build_faiss_index(model, database_root, index_dir, device)

    # Run evaluation
    results = evaluate(model, benchmark, index, all_paths, device, query_root, database_root)
    
    # Save results summary
    results_summary = {
        'model': checkpoint_path,
        'benchmark': benchmark_path,
        'metrics': results,
        'config': {
            'embedding_dim': 256,
            'device': device,
            'index_size': index.ntotal
        }
    }
    
    summary_path = "dualviewhair_evaluation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results_summary, f, indent=2)
    print(f"\\n💾 Evaluation summary saved to: {summary_path}")


if __name__ == '__main__':
    main()
