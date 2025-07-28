#!/usr/bin/env python3
"""Quick test script to verify your data structure."""

from pathlib import Path

def main():
    # Your data paths
    full_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/benchmark_processing/celeb10k"
    hair_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/hair_representation/hair_training_data/train/dummy"
    
    print("🧪 Testing Data Structure")
    print(f"Full images: {full_dir}")
    print(f"Hair images: {hair_dir}")
    
    full_path = Path(full_dir)
    hair_path = Path(hair_dir)
    
    # Check directories
    if not full_path.exists():
        print(f"❌ Full images directory missing")
        return
    if not hair_path.exists():
        print(f"❌ Hair images directory missing")
        return
    
    # Count files and matches
    full_images = list(full_path.glob("*.jpg"))
    hair_images = list(hair_path.glob("*_hair.png"))
    
    full_ids = {f.stem for f in full_images}
    hair_ids = {f.stem.replace('_hair', '') for f in hair_images}
    matched = full_ids.intersection(hair_ids)
    
    print(f"� Full images: {len(full_images)}")
    print(f"📁 Hair images: {len(hair_images)}")
    print(f"� Matched pairs: {len(matched)}")
    
    if len(matched) > 0:
        print(f"✅ Ready! Found {len(matched)} pairs.")
        print("\nNext: python scripts/simple_train.py --epochs 10")
    else:
        print("❌ No matching pairs found!")

if __name__ == "__main__":
    main()
