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
    def __init__(self, annotations_file, img_dir, transform=None, transform2=None, our_method=False):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.our_method = our_method
        self.transform2 = transform2

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
            anchor = self.transform(image)[0]
            pos1, pos2 = self.transform2(image)
            return {"anchor": anchor, "pos1": pos1, "pos2": pos2}
        else:
            image = self.transform(image)
            return image, label
