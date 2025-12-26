import argparse
from src.pretrain_engine import Trainer
from src.classification_engine import Classifier
from utils.utils import set_seed
import yaml
import os
from utils.transform import knn_transform
from utils.dataloader import CustomDataset
import torch
from torch.utils.data import DataLoader
from src.backbone import SupConResNet, SimCLR, MAE, DINOv2, SimMIM, SimCLR, SiameseIMViT, DenseCL, MSN
import torch
import torchvision
from torch import nn
from timm.models.vision_transformer import vit_base_patch16_224
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms import MAETransform
from lightly.transforms.dino_transform import DINOTransform
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    log_loss
)
from functools import partial
from src.main_backbone import SHAM

class LayerNorm(nn.LayerNorm):

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        return super(LayerNorm, self).forward(input.float())

def parse_args():
    parser = argparse.ArgumentParser(description="Self-supervised/Supervised Trainer Arguments")

    # Training config
    parser.add_argument('--save_path', type=str, default='classification_output_dir', help='Path to save model checkpoint')
    parser.add_argument('--size', type=int, default=224, help="Image size for training")
    parser.add_argument('--train_annotation', type=str, help='Path to training annotation file')
    parser.add_argument('--test_annotation', type=str, help='Path to testing annotation file')
    parser.add_argument('--img_dir', type=str, help='Path to image directory')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')

    # Model option
    parser.add_argument('--mode', type=str, default='simclr_supcon', choices=['mae', 'simclr', 'simclr_supcon', 'dinov2', 'simMIM', 'siaMIM', "SHAM", "DenseCL", "MSN"])
    parser.add_argument('--model', type=str, default='resnet18', choices = ['resnet18', 'resnet50', "vit_b_16"])
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: cuda or cpu')
    parser.add_argument('--SHAM_mode', type=str, default="embedding", choices=['embedding', 'reconstruction'])

    # Optional config
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--config', type=str, help='Optional path to YAML config file (overrides args)')
    parser.add_argument('--num_workers', type=int, default=4)

    # ViT settings
    parser.add_argument('--eval_type', default=None, type=str, choices=["knn", "linear_prob"])

    
    return parser.parse_args()

def merge_config_with_args(args):
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_file = yaml.safe_load(f)

        for key, value in config_file.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)
    return args

def main(args):

    
    # if args.mode == "simclr_supcon":
    #     mean = (0.5071, 0.4867, 0.4408) # cifar100
    #     std = (0.2675, 0.2565, 0.2761)
    #     train_transform = get_train_transform(args.size, mean, std)
    #     test_transform = get_test_transform(args.size, mean, std)
    # elif args.mode == "simclr" or args.mode=="our":
    #     train_transform = SimCLRTransform(input_size=224)
    #     test_transform = SimCLRTransform(input_size=224)   
    # elif args.mode == "mae":
    #     train_transform = MAETransform(input_size=224)
    #     test_transform = MAETransform(input_size=224)
    # elif args.mode == "dinov2":
    #     train_transform = DINOTransform()
    #     test_transform = DINOTransform()
    # elif args.mode == "simMIM":
    #     train_transform = MAETransform(input_size=224)
    #     test_transform = MAETransform(input_size=224)
    # elif args.mode == "siaMIM":
    #     state_dict = torch.load(args.checkpoint_path, map_location=args.device)
    #     ckpt_args = state_dict["args"]
    #     train_transform = DataAugmentationForSIM(args=ckpt_args)
    #     test_transform = DataAugmentationForSIM(args=ckpt_args)

    # if args.mode == "simclr_supcon":
    #     train_dataset = CustomDataset(args.train_annotation, args.img_dir, TwoCropTransform(train_transform))
    # else:
    #     train_dataset = CustomDataset(args.train_annotation, args.img_dir, train_transform)

    train_transform=knn_transform
    test_transform=knn_transform
    
    train_dataset = CustomDataset(args.train_annotation, args.img_dir, train_transform)
    test_dataset = CustomDataset(args.test_annotation, args.img_dir, test_transform)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers = args.num_workers)
    
    if args.mode == "simclr_supcon":
        model = SupConResNet(name=args.model, feat_dim=args.classes)
        state_dict = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True)
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded!")

    elif args.mode == "simclr":
        model = SimCLR(model=args.model)
        state_dict = torch.load(args.checkpoint_path, map_location=args.device, weights_only=True)
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded!")
    
    elif args.mode == "SHAM":
        model = SHAM()
        state_dict = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(state_dict['model_state_dict'])
        print("✅ Model weights loaded!")

    elif args.mode == "mae":
        vit = vit_base_patch16_224()
        model = MAE(vit)
        state_dict = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(state_dict['model_state_dict'])
        print("✅ Model weights loaded!")
    
    elif args.mode == "dinov2":
        model = DINOv2()
        state_dict = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded!")

    elif args.mode == "simMIM":
        vit = torchvision.models.vit_b_16(pretrained=False)
        model = SimMIM(vit)
        state_dict = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(state_dict)
        print("✅ Model weights loaded!")
    
    elif args.mode == "DenseCL":
        resnet = torchvision.models.resnet50()
        backbone = nn.Sequential(*list(resnet.children())[:-2])
        model = DenseCL(backbone)
        state_dict = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(state_dict['model_state_dict'])
        print("✅ Model weights loaded!")
    
    elif args.mode == "MSN":
        vit = torchvision.models.vit_b_16(pretrained=False)
        model = MSN(vit)
        state_dict = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(state_dict['model_state_dict'])
        print("✅ Model weights loaded!")
        
    elif args.mode == "siaMIM":
        state_dict = torch.load(args.checkpoint_path, map_location=args.device, weights_only=False)
        backbone = state_dict['model']
        ckpt_args = state_dict["args"]
        model = SiameseIMViT(
            patch_size=16, embed_dim=768, depth=12, num_heads=12,
            decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
            mlp_ratio=4, norm_layer=partial(LayerNorm, eps=1e-6), args=ckpt_args)
        model.load_state_dict(backbone)
        print("✅ Model weights loaded!")

    trainer = Classifier(model, train_loader, test_loader, args)
    if args.eval_type == "knn":
        trainer.knn_eval()
    elif args.eval_type == "linear_prob":
        trainer.linear_probe_eval()

if __name__ == "__main__":
    args = parse_args()
    args = merge_config_with_args(args)
    set_seed(args.seed)

    main(args)
