import numpy as np
import facer
import torch
import os
import json
import faiss
import torchvision
from src.backbone import SimCLR, SiameseIMViT, MSN
from src.main_backbone import SHAM2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from PIL import Image
import pickle
from torchvision import models, transforms
from torch import nn
import torchvision
import random
import shutil
from torchvision.transforms.functional import to_pil_image
from scipy.ndimage import binary_fill_holes
from functools import partial

class LayerNorm(nn.LayerNorm):

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        return super(LayerNorm, self).forward(input.float())



#device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cpu"

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def load_image(path):
    image = Image.open(path).convert("RGB")
    return transform(image).unsqueeze(0)

# ----------------------------
# Dataset Loader (Hair Region Only)
# ----------------------------
class HairDataset(Dataset):
    def __init__(self, items, data_dir):
        self.samples = [(item['id'], os.path.join(data_dir, item['hair_region_path'])) for item in items]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        image_id, image_path = self.samples[idx]
        try:
            image = Image.open(image_path).convert("RGB")
            return transform(image), image_path
        except:
            return torch.zeros(3, 224, 224), image_path


# ----------------------------
# Feature Extraction
# ----------------------------
@torch.no_grad()
def extract_feature(model, image_path, device, query_setting=False):
    # if query_setting:
    #     H, W, C = input_image.shape
    #     input_image = transform(to_pil_image(input_image))
    #     input_image = input_image.unsqueeze(0).to(device)
    # else:
    input_image = load_image(image_path).to(device)
    feat = model.extract_features(input_image).cpu().numpy()
    faiss.normalize_L2(feat)
    return feat

@torch.no_grad()
def build_index(model, items, data_dir, index_dir, device):
    dataset = HairDataset(items, data_dir)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=4)
    
    features, paths = [], []
    for images, img_paths in tqdm(loader, desc="Extracting features"):
        feats = model.extract_features(images.to(device)).cpu().numpy()
        features.append(feats)
        paths.extend(img_paths)
    
    feats_np = np.vstack(features)
    faiss.normalize_L2(feats_np)
    
    index = faiss.IndexFlatL2(feats_np.shape[1])
    index.add(feats_np)
    
    os.makedirs(index_dir, exist_ok=True)
    faiss.write_index(index, os.path.join(index_dir, "index_hair_region.index"))
    with open(os.path.join(index_dir, "paths_hair_region.pkl"), "wb") as f:
        pickle.dump(paths, f)

    return index, paths

@torch.no_grad()
def retrieve_random_queries_with_save(model, query_img, data_dir, index, all_paths, device, k=10, save_dir="retrieval_results"):
    
    # extract feature
    feat = extract_feature(model, query_img, device)
    print("Features: ", feat.shape)
    scores, idxs = index.search(feat, k)
    retrieved_paths = [all_paths[j] for j in idxs[0]]
    print("scores: ", scores)
    
    retrieved_ids = [os.path.splitext(os.path.basename(p))[0] for p in retrieved_paths]
    
    retrieved_imgs = []
    for id in retrieved_ids:
        if "hair" in id:
            img_path = os.path.join(data_dir, f"{id}.png")
        else:
            img_path = os.path.join(data_dir, f"{id}.jpg")
        print(img_path)
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img)
        gray = np.mean(arr, axis=-1)
        binary = gray > 0
        filled = binary_fill_holes(binary)
        background_mask = ~filled
        arr[background_mask] = [255, 255, 255]
        img = Image.fromarray(arr)
        retrieved_imgs.append(img)
    
    return retrieved_imgs, scores[0]

def sepia(input_img):
    sepia_filter = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ])
    sepia_img = input_img.dot(sepia_filter.T)
    sepia_img /= sepia_img.max()
    print(np.shape(sepia_img)) # (3454, 2301, 3)
    return sepia_img


def parsing_image(input_img):
    image_np = input_img
    print("=> ", np.shape(image_np))
    H, W, C = np.shape(image_np)
    image_tensor = facer.hwc2bchw(torch.from_numpy(image_np)).to(device)
    #image_tensor = torch.from_numpy(image_np.reshape(1, C, H, W)).to(device)
    print("image_tensor: ", image_tensor.shape)
    print("image_tensor: ", type(image_tensor))
    # face detector
    face_detector = facer.face_detector('retinaface/mobilenet', device=device)
    
    with torch.inference_mode():
        faces = face_detector(image_tensor)
    
    print("faces: ", faces)
    # face parsing
    face_parser = facer.face_parser('farl/lapa/448', device=device)
    with torch.inference_mode():
        faces = face_parser(image_tensor, faces)
    
    # segmentaiton
    seg_logits = faces["seg"]["logits"]  # [B, C, H, W]
    seg_probs = seg_logits.softmax(dim=1)
    seg_masks = seg_logits.argmax(dim=1).squeeze().cpu().numpy()  # (H, W)
    
    # hair mask
    hair_mask = (seg_masks == 10).astype(np.uint8) * 255  # (H, W)
    print("hair mask: ", np.shape(hair_mask))
    if len(np.shape(hair_mask)) == 3:
        hair_mask = hair_mask[1]
    
    # display
    background_black = np.ones_like(image_np, dtype=np.uint8) * 0  # black backgrond
    background_white = np.ones_like(image_np, dtype=np.uint8) * 255  # white backgrond
    hair_mask_3c = np.repeat(hair_mask[:, :, None], 3, axis=2)  # (H, W, 3)
    print("hair_masked_3c: ", np.shape(hair_mask_3c))
    hair_region = np.where(hair_mask_3c == 255, image_np, background_black)
    hair_display = np.where(hair_mask_3c == 255, image_np, background_white)
    print(np.shape(hair_region)) # (3454, 2301, 3)
    Image.fromarray(hair_region).save("app/hair_region.png")
    return hair_region, hair_display

def retrieve_img(input_img):
    pass

def general_pipeline(input_img, model):
    print("Using model: ", model)
    hair_region, hair_display = parsing_image(input_img)
    device="cpu"
    
    benchmark_path = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/hair_representation/test_retrieval/celebA_hairstyle_label_benchmark.json"
    benchmark_dir = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/high_resolution_FFHQ_labeled_celebA_hair_region"
    if model == "SimCLR":
        index_dir = "./app/faiss_index_simclr_hairstyle"
        checkpoint_path = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/HairPretraining/output_dir/simclr_resnet50/model_ckpt_latest.pth"
        model = SimCLR(model="resnet50").to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False)['model_state_dict'])
        model.eval()
    elif model == "SHAM":
        index_dir = "./app/faiss_index_SHAM_hairstyle"
        checkpoint_path = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/HairPretraining/output_dir/vast_ai_20_warm_up_simclr_resnet50_neg_sample_supervised_mse_static_alpha/model_ckpt_299.pth"
        model = SHAM2(model="resnet50").to(device)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=False))
        model.eval()
    elif model == "SiaMIM":
        index_dir = "./app/faiss_index_siaMIM_hairstyle"
        checkpoint_path = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/hair_representation/Siamese-Image-Modeling/output_dir/sim/checkpoint-299.pth"
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        backbone = state_dict['model']
        ckpt_args = state_dict["args"]
        model = SiameseIMViT(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(LayerNorm, eps=1e-6), args=ckpt_args)
        model = model.to(device)
        model.load_state_dict(backbone)
        model.eval()
    elif model=="MSN":
        index_dir = "./app/faiss_index_MSN_hairstyle"
        checkpoint_path = "/mnt/mmlab2024nas/thanhnd_student/QuocAnh/FCIR/Baselines/Composed-Image-Retrieval/HairPretraining/output_dir/MSN_vit_b_16/model_ckpt_latest.pth"
        vit = torchvision.models.vit_b_16(pretrained=False)
        model = MSN(vit)
        model.to(device)
        state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(state_dict['model_state_dict'])
        model.eval()
    print("✅ Model weights loaded!")
        
    
    
    with open(benchmark_path, "r") as f:
        data = json.load(f)
        #print("data: ", data['gallery_images'])
        
    
    
    if os.path.exists(os.path.join(index_dir, "index_hair_region.index")):
        index = faiss.read_index(os.path.join(index_dir, "index_hair_region.index"))
        with open(os.path.join(index_dir, "paths_hair_region.pkl"), "rb") as f:
            all_paths = pickle.load(f)
    else:
        index, all_paths = build_index(model, data['gallery_images'], benchmark_dir, index_dir, device)
    
    retrieved_imgs, scores = retrieve_random_queries_with_save(
                        model=model,
                        query_img="app/hair_region.png",
                        data_dir=benchmark_dir,
                        index=index,
                        all_paths=all_paths,
                        device=device,
                        k=5,
                        save_dir=f"retrieval_results/our_ablation_fixed_hard"
                    )
    
    return hair_display, retrieved_imgs[0], retrieved_imgs[1], retrieved_imgs[2], retrieved_imgs[3], retrieved_imgs[4], scores[0], scores[1], scores[2], scores[3], scores[4]
    #return hair_display, retrieved_imgs, scores