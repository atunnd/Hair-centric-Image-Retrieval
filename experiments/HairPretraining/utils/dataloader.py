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
from .transform import HairZoomTransform 

class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, transform2=None, our_method=False, multi_view=False):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.our_method = our_method
        self.transform2 = transform2
        if our_method:
            self.scale_transform = HairZoomTransform()
        self.multi_view = multi_view

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, img_name)

        try:
            img_bytes = read_file(img_path)  # Đọc file thành bytes
            image = decode_image(img_bytes, mode=torchvision.io.ImageReadMode.RGB)  # Giữ nguyên RGB

            image = to_pil_image(image)  # Nếu cần dùng PIL.Image
            
        except Exception as e:
            print(f"[WARNING] Failed to load image {img_path}: {e}")

        if self.our_method:
            image = self.scale_transform(image)
            anchor = self.transform(image)[0]
            pos1, pos2 = self.transform2(image)
            if self.multi_view:
                pos3, pos4 = self.transform2(image)
                return {"anchor": anchor, "pos1": pos1, "pos2": pos2, "pos3": pos3, "pos4": pos4}
            else:
                return {"anchor": anchor, "pos1": pos1, "pos2": pos2}
        else:
            image = self.transform(image)
            return image, label

import torch
from torchvision import transforms
import random
import numpy as np
from PIL import ImageFilter

# Gaussian Blur giống paper SimCLR
class GaussianBlur(object):
    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return x.filter(ImageFilter.GaussianBlur(radius=sigma))


def get_simclr_transform(image_size=224):
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        
        # Color jitter mạnh như SimCLR
        transforms.RandomApply([
            transforms.ColorJitter(
                brightness=0.8,
                contrast=0.8,
                saturation=0.8,
                hue=0.2
            )
        ], p=0.8),

        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([GaussianBlur()], p=0.5),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
