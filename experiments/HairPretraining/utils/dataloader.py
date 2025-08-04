import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image


class CustomDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, transform_target=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.transform_target = transform_target

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        label = self.img_labels.iloc[idx, 1]
        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
        if self.transform_target:
            label = self.transform_target

        return image, label
