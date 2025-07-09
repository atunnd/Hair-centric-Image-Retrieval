import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
from PIL import Image
from models import models_vit 
from sklearn.metrics.pairwise import cosine_similarity
import os
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle
from models.vit_face import ViT_face
from models.vits_face import ViTs_face 


class FaceEncoder:
    """Face Image Encoder using Vision Transformer for feature extraction and retrieval"""
    
    def __init__(self, ckpt_path, model_name="VIT", device=None):
        """
        Initialize the FaceEncoder
        
        Args:
            ckpt_path (str): Path to model checkpoint
            model_name (str): Model architecture name
            device (str): Device to use ('cuda', 'cpu', or None for auto-detect)
        """
        self.ckpt_path = ckpt_path
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and feature extractor
        self.model = None
        self.feature_extractor = None
        self.transform = self._get_transform()
        
        # Load model
        self._load_model()
    
    def _get_transform(self):
        """Get image transformation pipeline"""
        return transforms.Compose([
            transforms.Resize(112, interpolation=3),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _build_model(self):
        """Build the vision transformer model"""
        if self.model_name == "VIT":
            model = ViT_face(
            image_size=112,
            patch_size=8,
            loss_type='CosFace',
            GPU_ID= 'cuda:0',
            num_class=93431,
            dim=512,
            depth=20,
            heads=8,
            mlp_dim=2048,
            dropout=0.1,
            emb_dropout=0.1,
            pool='cls'
        )
        
        return model
    
    def _load_checkpoint(self, model):
        """Load model checkpoint"""
        msg = model.load_state_dict(torch.load(self.ckpt_path))
        print("Model loading message:", msg)
        return model
    
    def _load_model(self):
        """Load and initialize the model"""
        print(f"Building model: {self.model_name}")
        self.model = self._build_model()
        self.model = self._load_checkpoint(self.model)
        self.model.to(self.device)
        self.model.eval()
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(self.model) #@323232
    
    def extract_features(self, images):
        """
        Extract features from input images
        
        Args:
            images (torch.Tensor): Batch of images [B, C, H, W]
            
        Returns:
            torch.Tensor: Extracted features [B, feature_dim]
        """
        with torch.no_grad():
            features = self.feature_extractor.extract_features(images)
            return features
    
    def extract_dataset_features(self, data_path, batch_size=64, num_workers=8, save_dir="embeddings"):
        """
        Extract features from entire dataset and save embeddings
        
        Args:
            data_path (str): Path to dataset directory
            batch_size (int): Batch size for processing
            num_workers (int): Number of workers for data loading
            save_dir (str): Directory to save embeddings
            
        Returns:
            tuple: (embeddings, image_paths)
        """
        print(f"Loading dataset from: {data_path}")
        dataset = datasets.ImageFolder(data_path, transform=self.transform)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        all_embeddings = []
        all_paths = []
        
        with torch.no_grad():
            for imgs, _ in tqdm(loader, desc="Extracting embeddings"):
                imgs = imgs.to(self.device)
                features = self.extract_features(imgs)
                all_embeddings.append(features.cpu().numpy())
                
                start_idx = len(all_paths)
                all_paths.extend([dataset.samples[i][0] for i in range(start_idx, start_idx + imgs.size(0))])
        
        all_embeddings = np.concatenate(all_embeddings, axis=0)
        
        # Save embeddings and paths
        os.makedirs(save_dir, exist_ok=True)
        np.save(os.path.join(save_dir, "embeddings.npy"), all_embeddings)
        with open(os.path.join(save_dir, "image_paths.txt"), "w") as f:
            for path in all_paths:
                f.write(path + "\n")
        
        print(f"Saved {all_embeddings.shape[0]} embeddings and paths to {save_dir}")
        return all_embeddings, all_paths
    
    def load_embeddings(self, save_dir):
        """
        Load saved embeddings and paths
        
        Args:
            save_dir (str): Directory containing saved embeddings
            
        Returns:
            tuple: (embeddings, image_paths)
        """
        embeddings = np.load(os.path.join(save_dir, "embeddings.npy"))
        with open(os.path.join(save_dir, "image_paths.txt"), "r") as f:
            paths = [line.strip() for line in f.readlines()]
        return embeddings, paths
    
    def check_embeddings_exist(self, save_dir):
        """Check if embeddings exist in the given directory"""
        embeddings_file = os.path.join(save_dir, "embeddings.npy")
        paths_file = os.path.join(save_dir, "image_paths.txt")
        return os.path.exists(embeddings_file) and os.path.exists(paths_file)
    
    def encode_single_image(self, image_path):
        """
        Encode a single image
        
        Args:
            image_path (str): Path to the image
            
        Returns:
            np.ndarray: Image embedding
        """
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        embedding = self.extract_features(image_tensor).cpu().numpy()[0]
        return embedding
    
    def retrieve_similar_images(self, query_embedding, all_embeddings, all_paths, top_k=5):
        """
        Retrieve similar images based on cosine similarity
        
        Args:
            query_embedding (np.ndarray): Query image embedding
            all_embeddings (np.ndarray): All database embeddings
            all_paths (list): All image paths
            top_k (int): Number of top similar images to retrieve
            
        Returns:
            list: List of dictionaries with 'path' and 'similarity' keys
        """
        similarities = cosine_similarity([query_embedding], all_embeddings)[0]
        top_indices = np.argsort(similarities)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({'path': all_paths[idx], 'similarity': similarities[idx]})
        return results


class FeatureExtractor:
    """Feature extractor wrapper for the model"""
    
    def __init__(self, model):
        self.model = model
        self.model.eval()
    
    def extract_features(self, x):
        """Extract features from input tensor"""
        with torch.no_grad():
            cls_token = self.model.forward(x)
            return cls_token  # CLS token


class FaceRetrievalVisualizer:
    """Visualization utilities for hair retrieval results"""
    
    def __init__(self, vis_save_dir="visualizations"):
        self.vis_save_dir = vis_save_dir
        os.makedirs(vis_save_dir, exist_ok=True)
    
    def load_image_for_vis(self, image_path, target_size=(224, 224)):
        """Load image for visualization without normalization"""
        # display_path = full_face_dir + image_path.split("/")[-1].replace("_hair.png", ".jpg")
        display_path = image_path
        image = Image.open(display_path).convert('RGB')
        image = image.resize(target_size, Image.Resampling.LANCZOS)
        return np.array(image)
    
    def create_retrieval_visualization(self, query_path, results, save_path, top_k=5):
        """Create and save a visualization of retrieval results"""
        fig, axes = plt.subplots(1, top_k + 1, figsize=(4 * (top_k + 1), 4))
        
        # Load and display query image
        query_img = self.load_image_for_vis(query_path)
        axes[0].imshow(query_img)
        axes[0].set_title(f'Query Image\n{os.path.basename(query_path)}', fontsize=10)
        axes[0].axis('off')
        
        # Add border around query image
        rect = Rectangle((0, 0), query_img.shape[1], query_img.shape[0], 
                        linewidth=3, edgecolor='red', facecolor='none')
        axes[0].add_patch(rect)
        
        # Load and display retrieved images
        for i, result in enumerate(results):
            img = self.load_image_for_vis(result['path'])
            axes[i + 1].imshow(img)
            similarity = result['similarity']
            filename = os.path.basename(result['path'])
            axes[i + 1].set_title(f'Rank {i+1}\n{filename}\nSim: {similarity:.3f}', fontsize=10)
            axes[i + 1].axis('off')
            
            # Add border with color based on similarity
            color = plt.cm.RdYlGn(similarity)  # Red to green colormap
            rect = Rectangle((0, 0), img.shape[1], img.shape[0], 
                            linewidth=2, edgecolor=color, facecolor='none')
            axes[i + 1].add_patch(rect)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved visualization to: {save_path}")
    
    def create_combined_visualization(self, all_visualizations, save_path, top_k=5):
        """Create a combined visualization of all query results"""
        num_queries = len(all_visualizations)
        fig, axes = plt.subplots(num_queries, top_k + 1, figsize=(4 * (top_k + 1), 4 * num_queries))
        
        # Handle single query case
        if num_queries == 1:
            axes = axes.reshape(1, -1)
        
        for i, (query_path, results) in enumerate(all_visualizations):
            # Load and display query image
            query_img = self.load_image_for_vis(query_path)
            axes[i, 0].imshow(query_img)
            axes[i, 0].set_title(f'Query {i+1}\n{os.path.basename(query_path)}', fontsize=10)
            axes[i, 0].axis('off')
            
            # Add border around query image
            rect = Rectangle((0, 0), query_img.shape[1], query_img.shape[0], 
                            linewidth=3, edgecolor='red', facecolor='none')
            axes[i, 0].add_patch(rect)
            
            # Load and display retrieved images
            for j, result in enumerate(results):
                img = self.load_image_for_vis(result['path'])
                axes[i, j + 1].imshow(img)
                similarity = result['similarity']
                filename = os.path.basename(result['path'])
                axes[i, j + 1].set_title(f'Rank {j+1}\n{filename}\nSim: {similarity:.3f}', fontsize=10)
                axes[i, j + 1].axis('off')
                
                # Add border with color based on similarity
                color = plt.cm.RdYlGn(similarity)  # Red to green colormap
                rect = Rectangle((0, 0), img.shape[1], img.shape[0], 
                                linewidth=2, edgecolor=color, facecolor='none')
                axes[i, j + 1].add_patch(rect)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved combined visualization to: {save_path}")
    
    def visualize_multiple_queries(self, hair_encoder, embeddings, paths, num_queries=5, top_k=5, random_seed=42):
        """Visualize retrieval results for multiple random query images"""
        print(f"\nGenerating visualizations for {num_queries} random query images...")
        
        # Set random seed
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Randomly select query images
        num_images = len(paths)
        query_indices = random.sample(range(num_images), min(num_queries, num_images))
        
        # Store all visualizations for combined image
        all_visualizations = []
        
        for i, query_idx in enumerate(query_indices):
            query_path = paths[query_idx]
            print(f"\nProcessing query {i+1}/{len(query_indices)}: {os.path.basename(query_path)}")
            
            # Get query embedding
            query_embedding = embeddings[query_idx]
            
            # Retrieve similar images
            results = hair_encoder.retrieve_similar_images(query_embedding, embeddings, paths, top_k=top_k)
            
            # Store for combined visualization
            all_visualizations.append((query_path, results))
            
            # Create and save individual visualization
            # vis_filename = f"retrieval_query_{i+1}_{os.path.splitext(os.path.basename(query_path))[0]}.png"
            # vis_path = os.path.join(self.vis_save_dir, vis_filename)
            # self.create_retrieval_visualization(query_path, results, vis_path, top_k)
            
            # Print results
            print(f"Top {top_k} similar images:")
            for j, res in enumerate(results):
                print(f"  {j+1}. {os.path.basename(res['path'])} (similarity: {res['similarity']:.4f})")
        
        # Create combined visualization
        combined_vis_path = os.path.join(self.vis_save_dir, "combined_retrieval_results_face.png")
        self.create_combined_visualization(all_visualizations, combined_vis_path, top_k)
