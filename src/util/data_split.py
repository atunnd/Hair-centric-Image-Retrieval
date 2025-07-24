import os
import argparse
import shutil
from tqdm import tqdm  # Progress bar

def main():
    parser = argparse.ArgumentParser(description="Split images into folders of 5000 images each")
    parser.add_argument('--input_dir', type=str, help='Directory containing training data')
    parser.add_argument('--output_dir', type=str, help='Directory for target data')
    args = parser.parse_args()

    # Lấy danh sách file ảnh
    img_list = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    total_images = len(img_list)

    # Tính số thư mục cần tạo
    num_folders = (total_images + 4999) // 5000  # Làm tròn lên

    # Tạo thư mục đích
    list_folder = []
    for i in range(num_folders):
        folder_name = os.path.join(args.output_dir, f"hair_{i}")
        os.makedirs(folder_name, exist_ok=True)
        list_folder.append(folder_name)

    # Copy ảnh với progress bar
    for i in range(num_folders):
        lower_idx = i * 5000
        upper_idx = min(lower_idx + 5000, total_images)
        for img in tqdm(img_list[lower_idx:upper_idx], desc=f"Copying to hair_{i}"):
            img_name = os.path.splitext(img)[0]
            src = os.path.join(args.input_dir, img)
            dst = os.path.join(list_folder[i], f"{img_name}.png")  # convert to .png (optional)
            shutil.copy(src, dst)

if __name__ == "__main__":
    main()
