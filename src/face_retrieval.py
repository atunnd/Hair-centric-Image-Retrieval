import os
import argparse
import random
import numpy as np
from models.face_encoder import FaceEncoder, FaceRetrievalVisualizer


def parse_args():
    """Parse command line arguments for face retrieval inference"""
    parser = argparse.ArgumentParser(description='Face Image Retrieval Inference')
    
    # Model configuration
    parser.add_argument('--ckpt_path', type=str, default="weights/face_encoder/Backbone_VIT_Epoch_2_Batch_20000_Time_2021-01-12-16-48_checkpoint.pth",
                        help='Path to model checkpoint')
    parser.add_argument('--model_name', type=str, default="VIT",
                        choices=["VIT", "VITs"],
                        help='Model architecture to use')
    
    # Data configuration
    parser.add_argument('--data_path', type=str, 
                        default="/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/faceLearning/data/train",
                        help='Path to training data directory')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')
    
    # Device and output configuration
    parser.add_argument('--device', type=str, default=None,
                        choices=['cuda', 'cpu'],
                        help='Device to use (cuda/cpu). If None, auto-detect')
    parser.add_argument('--embed_save_dir', type=str, default="save/embeddings",
                        help='Directory to save embeddings')
    
    # Retrieval configuration
    parser.add_argument('--query_image', type=str, default=None,
                        help='Path to query image for retrieval (if None, use first image from dataset)')
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of top similar images to retrieve')
    
    # Visualization configuration
    parser.add_argument('--num_queries', type=int, default=5,
                        help='Number of random query images to use for visualization')
    parser.add_argument('--save_visualization', action='store_true',
                        help='Save retrieval visualizations')
    parser.add_argument('--vis_save_dir', type=str, default="save/visualizations",
                        help='Directory to save visualizations')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Random seed for reproducible query selection')
    
    # Action flags
    parser.add_argument('--extract_only', action='store_true',
                        help='Only extract embeddings, skip retrieval')
    parser.add_argument('--retrieve_only', action='store_true',
                        help='Only perform retrieval, skip embedding extraction')
    parser.add_argument('--force_extract', action='store_true',
                        help='Force re-extraction of embeddings even if they exist')
    
    return parser.parse_args()


def print_config(args):
    """Print configuration summary"""
    print("=" * 60)
    print("FACE IMAGE RETRIEVAL INFERENCE")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  - Checkpoint: {args.ckpt_path}")
    print(f"  - Model: {args.model_name}")
    print(f"  - Data path: {args.data_path}")
    print(f"  - Batch size: {args.batch_size}")
    print(f"  - Device: {args.device if args.device else 'auto-detect'}")
    print(f"  - Embed save dir: {args.embed_save_dir}")
    print(f"  - Top K: {args.top_k}")
    if args.save_visualization:
        print(f"  - Visualization: Enabled (save to {args.vis_save_dir})")
        print(f"  - Number of queries: {args.num_queries}")
        print(f"  - Random seed: {args.random_seed}")
    print("=" * 60)


def extract_embeddings(face_encoder, args):
    """Extract embeddings from dataset"""
    print("Extracting embeddings from dataset...")
    embeddings, paths = face_encoder.extract_dataset_features(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        save_dir=args.embed_save_dir
    )
    return embeddings, paths


def load_embeddings(face_encoder, args):
    """Load existing embeddings"""
    print("Loading existing embeddings...")
    embeddings, paths = face_encoder.load_embeddings(args.embed_save_dir)
    return embeddings, paths


def single_query_retrieval(face_encoder, embeddings, paths, args):
    """Perform single query retrieval"""
    print("\n" + "=" * 60)
    print("SINGLE QUERY RETRIEVAL")
    print("=" * 60)
    
    # Determine query image
    if args.query_image:
        query_img_path = args.query_image
    else:
        query_img_path = paths[0]
        print(f"No query image specified, using first image from dataset: {query_img_path}")
    
    # Encode query image
    print(f"Encoding query image: {query_img_path}")
    query_embedding = face_encoder.encode_single_image(query_img_path)
    
    # Retrieve similar images
    results = face_encoder.retrieve_similar_images(query_embedding, embeddings, paths, top_k=args.top_k)
    
    # Print results
    print(f"\nTop {args.top_k} similar images to: {os.path.basename(query_img_path)}")
    print("-" * 60)
    for i, res in enumerate(results):
        print(f"{i+1}. {os.path.basename(res['path'])} (similarity: {res['similarity']:.4f})")
    
    return query_img_path, results


def visualize_retrieval(face_encoder, embeddings, paths, args):
    """Generate retrieval visualizations"""
    print("\n" + "=" * 60)
    print("GENERATING RETRIEVAL VISUALIZATIONS")
    print("=" * 60)
    
    # Initialize visualizer
    visualizer = FaceRetrievalVisualizer(vis_save_dir=args.vis_save_dir)
    
    # Generate visualizations
    visualizer.visualize_multiple_queries(
        face_encoder=face_encoder,
        embeddings=embeddings,
        paths=paths,
        num_queries=args.num_queries,
        top_k=args.top_k,
        random_seed=args.random_seed
    )


def main():
    """Main inference pipeline"""
    # Parse arguments
    args = parse_args()
    
    # Print configuration
    print_config(args)
    
    # Initialize faceEncoder
    print("Initializing FaceEncoder...")
    face_encoder = FaceEncoder(
        ckpt_path=args.ckpt_path,
        model_name=args.model_name,
        device=args.device
    )
    
    # Handle embedding extraction/loading
    should_extract = not args.retrieve_only and (args.force_extract or not face_encoder.check_embeddings_exist(args.embed_save_dir))
    
    if should_extract:
        if args.force_extract:
            print("Force extraction enabled.")
        else:
            print("Embeddings not found.")
        embeddings, paths = extract_embeddings(face_encoder, args)
    else:
        if not args.extract_only:
            embeddings, paths = load_embeddings(face_encoder, args)
    
    # Perform retrieval if not extract_only
    if not args.extract_only:
        if args.save_visualization:
            # Generate visualizations for multiple random queries
            visualize_retrieval(face_encoder, embeddings, paths, args)
        else:
            # Single query retrieval
            single_query_retrieval(face_encoder, embeddings, paths, args)
    
    print("\n" + "=" * 60)
    print("INFERENCE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()