import argparse
from src.pretrain_engine import Trainer
from src.classification_engine import Classifier
from utils.utils import set_seed
import yaml
import os
from utils.transform import get_test_transform, get_train_transform, TwoCropTransform
from utils.dataloader import CustomDataset
import torch
from torch.utils.data import DataLoader
from src.backbone import SupConResNet, SimCLR, MAE, DINO, SimMIM, OriginSimCLR
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
from utils.transform import DataAugmentationForSIMWithMask
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score,
    log_loss
)


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
    parser.add_argument('--mode', type=str, default='simclr_supcon', choices=['mae', 'simclr', 'simclr_supcon', 'dino', 'simMIM'])
    parser.add_argument('--model', type=str, default='resnet18', choices = ['resnet18', 'resnet50', "vit_b_16"])
    parser.add_argument('--checkpoint_path', type=str, default=None)
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: cuda or cpu')
    parser.add_argument('--our_method', type=bool, default=False)

    # Optional config
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--config', type=str, help='Optional path to YAML config file (overrides args)')
    parser.add_argument('--num_workers', type=int, default=4)

    # ViT settings
    parser.add_argument('--fusion_type', default="mlp", type=str, choices=["mlp", "transformer"])

    
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

    
    if args.mode == "simclr_supcon":
        mean = (0.5071, 0.4867, 0.4408) # cifar100
        std = (0.2675, 0.2565, 0.2761)
        train_transform = get_train_transform(args.size, mean, std)
        test_transform = get_test_transform(args.size, mean, std)
    elif args.mode == "simclr":
        if "vit" in args.model:
            print("vit transformation")
            aug_args = type('', (), {})()
            aug_args.input_size = args.size
            aug_args.crop_min = args.crop_min
            aug = DataAugmentationForSIMWithMask(aug_args)
            train_transform = aug
            test_transform = aug
        else:
            train_transform = SimCLRTransform(input_size=224)
            test_transform = SimCLRTransform(input_size=224)   
    elif args.mode == "mae":
        train_transform = MAETransform(input_size=224)
        test_transform = MAETransform(input_size=224)
    elif args.mode == "dino":
        train_transform = DINOTransform()
        test_transform = DINOTransform()
    elif args.mode == "simMIM":
        train_transform = MAETransform(input_size=224)
        test_transform = MAETransform(input_size=224)

    if args.mode == "simclr_supcon":
        train_dataset = CustomDataset(args.train_annotation, args.img_dir, TwoCropTransform(train_transform))
    else:
        train_dataset = CustomDataset(args.train_annotation, args.img_dir, train_transform)

    test_dataset = CustomDataset(args.test_annotation, args.img_dir, test_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers = args.num_workers)
    
    if args.mode == "simclr_supcon":
        model = SupConResNet(name=args.model, feat_dim=args.classes)
    elif args.mode == "simclr":
        if args.model == "resnet18":
            backbone = torchvision.models.resnet18()
            output_dim=128
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif args.model == "resnet50":
            backbone = torchvision.models.resnet50()
            output_dim=1024
            backbone = nn.Sequential(*list(backbone.children())[:-1])
        elif args.model == 'vit_b_16':
            backbone=None
            output_dim= 512

        if "vit" in str(args.model):
            model = SimCLR(backbone, args.model, attention_pooling = False)
        else:
            model = OriginSimCLR(backbone, args.model, attention_pooling = False)

    elif args.mode == "mae":
        vit = vit_base_patch16_224()
        model = MAE(vit)
    
    # elif args.mode == "dino":
    #     #backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=False, source="github")
    #     backbone = vits.vit_base(patch_size=16)
    #     input_dim = backbone.embed_dim
    #     model = DINO(backbone, input_dim)

    elif args.mode == "simMIM":
        vit = torchvision.models.vit_b_16(pretrained=False)
        model = SimMIM(vit)
    
    state_dict = torch.load(args.checkpoint_path, map_location=args.device)
    model.load_state_dict(state_dict)
    print("✅ Model weights loaded!")

    trainer = Classifier(model, train_loader, test_loader, args)
    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    args = merge_config_with_args(args)
    set_seed(args.seed)

    main(args)
