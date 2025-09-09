import torch
from torchvision import transforms, datasets
from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_image
import torch
import torchvision.transforms as T


positive_transform = T.Compose([
    T.RandomRotation(degrees=(-15, 15)),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5)),
])

negative_transform = T.Compose([     # Zoom nhẹ hơn positive
    T.RandomHorizontalFlip(p=0.5),                     # Flip như positive
    T.ColorJitter(brightness=0.1, contrast=0.1,         # Thay đổi màu rất nhẹ
                  saturation=0.1, hue=0.02),
])


def get_train_transform(size, mean, std):
    """
    Trả về chuỗi transform dùng cho ảnh huấn luyện.
    Args:
        opt: đối tượng chứa các tham số như opt.size, opt.mean, opt.std
    Returns:
        train_transform: torchvision.transforms.Compose
    """
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        #transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.Resize(size),
        transforms.ToTensor(),
        normalize
    ])

    return train_transform

def get_test_transform(size, mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    return transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        normalize
    ])

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

import torch
import torch.nn.functional as F

import torch
import torch.nn.functional as F

class PositiveMaskingTransform:
    """
    Custom transform to apply masking on a batch of positive images (tensors) after SimCLR transforms.
    Assumes input is a torch.Tensor (B, C, H, W) in [0,1] range, with black background (0).
    Masks 10-20% of hair-containing patches by setting them to 0, independently for each image in the batch.
    
    Args:
    - patch_size: int, size of each patch (e.g., 32 for 32x32 patches)
    - mask_ratio_range: tuple, range for random mask ratio (e.g., (0.1, 0.2) for 10-20%)
    - threshold: float, mean pixel value to consider a patch as containing hair
    """
    def __init__(self, patch_size=32, mask_ratio_range=(0.1, 0.2), threshold=0.01):
        self.patch_size = patch_size
        self.mask_ratio_range = mask_ratio_range
        self.threshold = threshold

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        """
        Apply masking to the input batch tensor.
        
        Args:
        - images: torch.Tensor (B, C, H, W), the batch of positive images after SimCLR transforms
        
        Returns:
        - masked_images: torch.Tensor (B, C, H, W), the masked versions
        """
        if not isinstance(images, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")
        
        B, C, H, W = images.shape
        num_patches_h = H // self.patch_size
        num_patches_w = W // self.patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Unfold to extract patches: (B, C, num_h, num_w, patch_h, patch_w)
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        
        # Reshape for mean calculation: (B, C, num_patches, patch_h, patch_w)
        patches = patches.contiguous().view(B, C, num_patches, self.patch_size, self.patch_size)
        
        # Identify hair patches: mean > threshold, averaged over C, patch_h, patch_w
        patch_means = patches.mean(dim=[1, 3, 4])  # (B, num_patches)
        hair_masks = patch_means > self.threshold  # (B, num_patches)
        
        masked_images = images.clone()
        
        for b in range(B):
            hair_indices = torch.nonzero(hair_masks[b]).squeeze(-1)  # Indices for this batch item
            if len(hair_indices) == 0:
                continue  # No hair, skip
            
            # Random mask ratio for this image
            mask_ratio = torch.empty(1, device=images.device).uniform_(*self.mask_ratio_range).item()
            num_mask = int(len(hair_indices) * mask_ratio)
            if num_mask == 0:
                continue
            
            # Random select indices to mask
            mask_indices = hair_indices[torch.randperm(len(hair_indices), device=images.device)[:num_mask]]
            
            # Mask selected patches (set to 0)
            for idx in mask_indices:
                ph = idx // num_patches_w
                pw = idx % num_patches_w
                masked_images[b, :, ph*self.patch_size:(ph+1)*self.patch_size, pw*self.patch_size:(pw+1)*self.patch_size] = 0.0
        
        return masked_images

# Example usage in SimCLR pipeline:
# Assume simclr_transform is your Compose for SimCLR (returns tensor)
# In data loader or forward:
# positives = simclr_transform(batch_pil_images)  # torch.Tensor (B, C, H, W), e.g., [256, 3, 224, 224]
# masking_transform = PositiveMaskingTransform()
# masked_positives = masking_transform(positives)
# Example usage in SimCLR pipeline:
# Assume simclr_transform is your Compose for SimCLR (returns tensor)
# In data loader or forward:
# positive = simclr_transform(pil_image)  # torch.Tensor (C, H, W)
# masking_transform = PositiveMaskingTransform()
# masked_positive = masking_transform(positive)


import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import torch
import math
import random
import numpy as np
from PIL import Image
import einops


# ===== Custom Flip =====
class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def forward(self, img, flip=None):
        if flip is None:
            flip = torch.rand(1) < self.p
        if flip:
            return F.hflip(img), True
        return img, False


# ===== Custom Crop =====
class SingleRandomResizedCrop(transforms.RandomResizedCrop):
    @staticmethod
    def get_params(img, scale, ratio):
        width, height = F.get_image_size(img)
        area = height * width

        log_ratio = torch.log(torch.tensor(ratio))
        for _ in range(10):
            target_area = area * torch.empty(1).uniform_(scale[0], scale[1]).item()
            aspect_ratio = torch.exp(torch.empty(1).uniform_(log_ratio[0], log_ratio[1])).item()

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if 0 < w <= width and 0 < h <= height:
                i = torch.randint(0, height - h + 1, size=(1,)).item()
                j = torch.randint(0, width - w + 1, size=(1,)).item()
                return i, j, h, w, width

        # fallback center crop
        in_ratio = float(width) / float(height)
        if in_ratio < min(ratio):
            w = width
            h = int(round(w / min(ratio)))
        elif in_ratio > max(ratio):
            h = height
            w = int(round(h * max(ratio)))
        else:
            w = width
            h = height
        i = (height - h) // 2
        j = (width - w) // 2
        return i, j, h, w, width

    def forward(self, img, i=None, j=None, h=None, w=None):
        if i is None or j is None or h is None or w is None:
            i, j, h, w, W = self.get_params(img, self.scale, self.ratio)
        else:
            W, _ = F.get_image_size(img)
        return F.resized_crop(img, i, j, h, w, self.size, self.interpolation), i, j, h, w, W


# ===== Helper: patchify mask =====
# def get_hair_region_idx(mask_t, patch_size=16):
#     B, H, W = 1, mask_t.shape[1], mask_t.shape[2]
#     nh, nw = H // patch_size, W // patch_size

#     mask_patches = einops.rearrange(
#         mask_t.unsqueeze(0),
#         "b 1 (nh ph) (nw pw) -> b (nh nw) (ph pw)",
#         ph=patch_size, pw=patch_size
#     )
#     has_hair = (mask_patches.sum(dim=-1) > 0)
#     hair_region_idx = torch.nonzero(has_hair[0], as_tuple=False).squeeze(1)
#     return hair_region_idx

import torch
import einops

def get_hair_region_idx(mask_t, patch_size=16):
    # mask_t: [batch_size, H, W], binary mask of hair region
    B, H, W = mask_t.shape
    nh, nw = H // patch_size, W // patch_size

    # Convert image mask to patch-level mask
    mask_patches = einops.rearrange(
        mask_t.unsqueeze(1),  # Add channel dim
        "b 1 (nh ph) (nw pw) -> b (nh nw) (ph pw)",
        ph=patch_size, pw=patch_size
    )  # [batch_size, num_patches, patch_size*patch_size]

    # Binary mask: 1 if patch contains hair (any pixel > 0), 0 otherwise
    has_hair = (mask_patches.sum(dim=-1) > 0).float()  # [batch_size, num_patches]
    return has_hair


# ===== Augmentation with mask =====
class DataAugmentationForSIMWithMask(object):
    def __init__(self, args):
        self.args = args

        self.random_resized_crop = SingleRandomResizedCrop(
            args.input_size, scale=(args.crop_min, 1.0), interpolation=3
        )
        self.random_flip = RandomHorizontalFlip()

        self.color_transform1 = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=1.0),
        ])

        self.color_transform2 = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))
            ], p=1.0),
        ])

        # ✅ Positive transform (split thành rotate + blur)
        self.pos_rotation = transforms.RandomRotation(degrees=(-45, 45))
        self.pos_blur = transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))

        self.format_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
        ])

    def __call__(self, image, mask):
        # === 1. Flip đồng bộ ===
        image1, flip1 = self.random_flip(image)
        image2, flip2 = self.random_flip(image)
        mask1, _ = self.random_flip(mask, flip=flip1)
        mask2, _ = self.random_flip(mask, flip=flip2)

        # === 2. Crop đồng bộ ===
        image1, i1, j1, h1, w1, W = self.random_resized_crop(image1)
        image2, i2, j2, h2, w2, W = self.random_resized_crop(image2)
        mask1, _, _, _, _, _ = self.random_resized_crop(mask1, i=i1, j=j1, h=h1, w=w1)
        mask2, _, _, _, _, _ = self.random_resized_crop(mask2, i=i2, j=j2, h=h2, w=w2)

        # === 3. Color aug chỉ cho ảnh ===
        color_image1 = self.color_transform1(image1)
        color_image2 = self.color_transform2(image2)

        # === 4. Positive transform từ image2 + mask2 ===
        angle = random.uniform(-15, 15)
        pos_image = F.rotate(image2, angle)
        pos_image = self.pos_blur(pos_image)  # chỉ blur image
        pos_mask = F.rotate(mask2, angle)     # mask chỉ rotate, không blur

        # === 5. To tensor ===
        image1_t = self.format_transform(color_image1)
        image2_t = self.format_transform(color_image2)
        pos_image_t = self.format_transform(pos_image)

        mask1_t = (F.to_tensor(mask1)[0:1, :, :] > 0).float()
        mask2_t = (F.to_tensor(mask2)[0:1, :, :] > 0).float()
        pos_mask_t = (F.to_tensor(pos_mask)[0:1, :, :] > 0).float()

        # === 6. Hair idx ===
        mask1_idx = get_hair_region_idx(mask1_t, patch_size=16)
        mask2_idx = get_hair_region_idx(mask2_t, patch_size=16)
        pos_mask_idx = get_hair_region_idx(pos_mask_t, patch_size=16)

        relative_flip = (flip1 and not flip2) or (flip2 and not flip1)

        return (image1_t, mask1_t, mask1_idx), \
               (image2_t, mask2_t, mask2_idx), \
               (pos_image_t, pos_mask_t, pos_mask_idx), \
               (i2 - i1) / h1, (j2 - j1) / w1, h2 / h1, w2 / w1, relative_flip, (W - j1 - j2) / w1