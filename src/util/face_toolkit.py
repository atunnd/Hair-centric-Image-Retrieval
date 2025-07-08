import torch
import facer
import torchvision.transforms as T
from PIL import Image
import os
from torchvision.utils import save_image
from multiprocessing import Pool, get_context, cpu_count
from tqdm import tqdm
import argparse

# Face parsing - these will be initialized in each worker process
face_parser = None
face_detector = None

def _init_worker():
    """Initialize worker process with face parsing models"""
    global face_parser, face_detector
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    face_parser = facer.face_parser('farl/lapa/448', device=device)
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)

def process_face_image(image_path, detector_model='retinaface/mobilenet', parser_model='farl/lapa/448', hair_class_index=10):
    """
    Process a face image to extract hair mask and original image.
    
    Args:
        image_path (str): Path to the input image
        detector_model (str): Face detector model name (default: 'retinaface/mobilenet')
        parser_model (str): Face parser model name (default: 'farl/lapa/448')
        hair_class_index (int): Class index for hair in segmentation (default: 10 for farl/lapa/448)
    
    Returns:
        tuple: (visualized_segmentation, visualized_segmentation_probs, hair_mask, original_image)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load and preprocess image
    image = facer.hwc2bchw(facer.read_hwc(image_path)).to(device=device)
    
    with torch.inference_mode():
        faces = face_detector(image)
        faces = face_parser(image, faces)
    
    # Process segmentation
    seg_logits = faces['seg']['logits']
    seg_probs = seg_logits.softmax(dim=1)
    n_classes = seg_probs.size(1)
    
    vis_seg_probs = seg_probs.argmax(dim=1).float() / n_classes * 255
    vis_img = vis_seg_probs.sum(0, keepdim=True)
    
    # Extract hair mask
    hair_mask = seg_probs[:, hair_class_index, :, :]
    hair_mask = (hair_mask > 0.5).float()
    hair_mask = hair_mask.sum(0, keepdim=True)
    hair_mask = (hair_mask > 0).float()

    return vis_img, vis_seg_probs, hair_mask, image

def save_hair_region(hair_mask, original_image, output_path, size=(224, 224)):
    """
    Extract hair region from original image using hair mask and save as RGB image.
    
    Args:
        hair_mask (torch.Tensor): Binary hair mask of shape [1, h, w]
        original_image (torch.Tensor): Original image of shape [1, 3, h, w]
        output_path (str): Path to save the hair region image
        size (tuple): Target size (width, height)
    """
    # Ensure hair mask is [1, h, w]
    hair_mask = hair_mask.squeeze(0)  # [h, w]
    
    # Convert original image to [3, h, w]
    original_image = original_image.squeeze(0)  # [3, h, w]
    
    # Apply mask: Keep hair pixels, set background to black (0)
    hair_mask = hair_mask.unsqueeze(0)  # [1, h, w]
    hair_region = original_image * hair_mask  # [3, h, w]
    
    save_image(hair_region, output_path, normalize=True)

def _process_single_image(args):
    """
    Process a single image to extract hair region
    
    Args:
        args: tuple of (image_path, output_path, hair_class_index, size)
    
    Returns:
        tuple: (success, message)
    """
    image_path, output_path, hair_class_index, size = args
    
    try:
        # Process the image
        vis_img, vis_seg_probs, hair_mask, original_image = process_face_image(
            image_path, hair_class_index=hair_class_index
        )
        
        # Save hair region
        save_hair_region(hair_mask, original_image, output_path, size)
        
        return True, f"Successfully processed {os.path.basename(image_path)}"
        
    except Exception as e:
        return False, f"Failed to process {os.path.basename(image_path)}: {e}"

def process_folder_multiprocess(input_dir, output_dir, hair_class_index=10, size=(224, 224), num_processes=None):
    """
    Process all images in a folder using multiple processes
    
    Args:
        input_dir (str): Directory containing input images
        output_dir (str): Directory to save hair region images
        hair_class_index (int): Hair class index for segmentation
        size (tuple): Target size for output images
        num_processes (int): Number of processes to use (None for auto)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = [f for f in os.listdir(input_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    
    if not image_files:
        print("No image files found in input directory!")
        return
    
    # Create tasks
    tasks = []
    for image_file in image_files:
        base_name = os.path.splitext(image_file)[0]
        image_path = os.path.join(input_dir, image_file)
        output_path = os.path.join(output_dir, f"{base_name}_hair.png")
        
        tasks.append((image_path, output_path, hair_class_index, size))
    
    print(f"Found {len(tasks)} images to process")
    
    # Set number of processes
    if num_processes is None:
        num_processes = min(cpu_count(), 4)  # Limit to 4 to avoid GPU memory issues
    
    print(f"Using {num_processes} processes")
    
    # Process images with multiprocessing
    with get_context("spawn").Pool(processes=num_processes, initializer=_init_worker) as pool:
        results = list(tqdm(
            pool.imap(_process_single_image, tasks), 
            total=len(tasks), 
            desc="Extracting hair regions"
        ))
    
    # Report results
    successful = [msg for success, msg in results if success]
    failed = [msg for success, msg in results if not success]
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {len(successful)} images")
    if failed:
        print(f"Failed to process: {len(failed)} images")
        for msg in failed[:5]:  # Show first 5 failures
            print(f"  {msg}")
        if len(failed) > 5:
            print(f"  ... and {len(failed) - 5} more")

def main():
    parser = argparse.ArgumentParser(description='Extract hair regions from face images using multiprocessing')
    parser.add_argument('--input_dir', type=str, default="/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/assets/samples",
                       help='Directory containing input face images')
    parser.add_argument('--output_dir', type=str, default="/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/assets/hair_region_only",
                       help='Directory to save hair region images')
    parser.add_argument('--hair_class_index', type=int, default=10,
                       help='Hair class index for segmentation (default: 10)')
    parser.add_argument('--size', type=int, nargs=2, default=[1024, 1024],
                       help='Target size for output images (width height)')
    parser.add_argument('--num_processes', type=int, default=1,
                       help='Number of processes to use (default: auto)')
    
    args = parser.parse_args()
    
    # Convert size to tuple
    size = tuple(args.size)
    
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Hair class index: {args.hair_class_index}")
    print(f"Target size: {size}")
    
    process_folder_multiprocess(
        args.input_dir, 
        args.output_dir, 
        args.hair_class_index, 
        size, 
        args.num_processes
    )

if __name__ == '__main__':
    main()