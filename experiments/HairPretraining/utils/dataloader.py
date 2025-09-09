import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torchvision
from torchvision.io import read_file, decode_image
from torchvision.transforms.functional import to_pil_image
import cv2
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, transform_target=None, original_img_dir=None, our_method=False):
        self.img_labels = pd.read_csv(annotations_file)[:20000]
        self.img_dir = img_dir
        self.transform = transform
        self.transform_target = transform_target
        self.aug = transform
        self.original_img_dir = original_img_dir
        self.our_method = our_method

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, img_name)
        img_name_origin = f"{img_name[:6]}.jpg"
        #img_origin_path = os.path.join(self.original_img_dir, img_name_origin)

        if self.our_method:
            try:
                image_orig = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)  # (H, W, 3)
                mask_orig  = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
                mask_orig  = np.stack([mask_orig, mask_orig, mask_orig], axis=-1)

                image_pil = Image.fromarray(image_orig)
                mask_pil  = Image.fromarray(mask_orig)

                (image1_t, mask1_t, mask1_idx), (image2_t, mask2_t, mask2_idx), (image3_t, mask3_t, mask3_idx), *_ = self.aug(image_pil, mask_pil)

            except Exception as e:
                print(f"[WARNING] Failed to load image {img_path}: {e}")
            return [[image1_t, image2_t, image3_t], [mask1_idx, mask2_idx, mask3_idx], label]

        else:
            try:
                img_bytes = read_file(img_path)  # Đọc file thành bytes
                image = decode_image(img_bytes, mode=torchvision.io.ImageReadMode.RGB)  # Giữ nguyên RGB

                image = to_pil_image(image)  # Nếu cần dùng PIL.Image
                
            except Exception as e:
                print(f"[WARNING] Failed to load image {img_path}: {e}")


            if self.transform:
                image = self.transform(image)
            if self.transform_target:
                label = self.transform_target

            dummy_mask = torch.zeros(1)

            return [image, dummy_mask, label]
        
        # samples = {
        #     "images": torch.stack([image1_t, image2_t, image3_t]),
        #     "mask_indices": torch.stack([mask1_idx, mask2_idx, mask3_idx])
        # }
