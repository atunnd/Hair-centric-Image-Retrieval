import argparse
from src.pretrain_engine import Trainer
from utils.utils import set_seed
import yaml
import os
from utils.transform import get_test_transform, get_train_transform, TwoCropTransform, SHAMTransform
from utils.dataloader import CustomDataset
import torch
from torch.utils.data import DataLoader
from src.backbone import SupConResNet, SimCLR, MAE, DINOv2, SimMIM
from src.main_backbone import SHAM
import torch
import torchvision
from torch import nn
from timm.models.vision_transformer import vit_base_patch16_224
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.transforms import MAETransform
from lightly.transforms.dino_transform import DINOTransform
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard import SummaryWriter



def parse_args():
    parser = argparse.ArgumentParser(description="Self-supervised/Supervised Trainer Arguments")

    # Training config
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: cuda or cpu')
    parser.add_argument('--device_id', type=int, default=0)
    parser.add_argument('--save_path', type=str, default='output_dir', help='Path to save model checkpoint')
    parser.add_argument('--size', type=int, default=224, help="Image size for training")
    parser.add_argument('--train_annotation', type=str, help='Path to training annotation file')
    parser.add_argument('--test_annotation', type=str, help='Path to testing annotation file')
    parser.add_argument('--img_dir', type=str, help='Path to image directory')
    parser.add_argument('--img_dir_origin', type=str, default=None, help="Path to original image")
    parser.add_argument('--continue_training',  action="store_true", help="Continue training")
    parser.add_argument('--checkpoint_folder', type=str, default=None, help="Path to checkpoint folder for resuming training")
    parser.add_argument('--training_settings', type=int, default=1, help="Training settings for SHAM", choices=[1,2,3,4])
    parser.add_argument('--full_face_training', action="store_true")
    parser.add_argument('--multi_view', action="store_true")
    parser.add_argument('--no_contrastive_loss', action="store_true")

    # optimization config
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default = 1e-4, help='Weight decay rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='1st moment')
    parser.add_argument('--beta2', type=float, default=0.999, help='2nd moment')

    # loss config
    parser.add_argument('--temp', type=float, default=0.5, help="temperature for loss function")

    # Model option
    parser.add_argument('--mode', type=str, default='simclr_supcon', choices=['mae', 'simclr', 'simclr_supcon', 'dinov2', 'simMIM', 'SHAM', 'S2R2'])
    parser.add_argument('--model', type=str, default='resnet18', choices = ['resnet18', 'resnet50', "vit_b_16"])
    parser.add_argument("--SHAM_mode", type=str, default="embedding", choices = ['reconstruction', 'embedding'])

    # Optional config
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--config', type=str, help='Optional path to YAML config file (overrides args)')
    parser.add_argument('--num_workers', type=int, default=4)

    # negative sampling
    parser.add_argument("--negative_sampling", action="store_true")
    parser.add_argument('--warm_up_epochs', default=20, type=int, help='Number of warmup epochs for negative sampling')
    parser.add_argument('--sampling_frequency', type=int, default=30, help="Frequency to sample hard negative")
    parser.add_argument('--ema', type=float, default=0.99)
    
    # retrieval setting
    parser.add_argument('--S2R2', action="store_true", help="Adding S2R2 regularization")

    # supcon setting
    parser.add_argument('--classes', default=128, type=int, help="Classes for sup con")

    # ViT settings
    parser.add_argument('--atn_pooling', default=False, type=bool, help='attention pooling for constrative learning')

    # augmentation settings
    parser.add_argument('--crop_min', default=0.2, type=float, help="crop min for random crop")

    # pooling settings
    parser.add_argument("--pooling_type", default="max_pooling", type=str, choices=["max_pooling", "mean_pooling", "GeM"])
    

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
        train_transform = SimCLRTransform(input_size=224)
        test_transform = SimCLRTransform(input_size=224)   
    elif args.mode == "mae":
        train_transform = MAETransform(input_size=224)
        test_transform = MAETransform(input_size=224)
    elif args.mode == "dinov2":
        train_transform = DINOTransform(
            global_crop_scale=(0.32, 1),
            local_crop_scale=(0.05, 0.32),
            n_local_views=8,)
        test_transform = DINOTransform(
            global_crop_scale=(0.32, 1),
            local_crop_scale=(0.05, 0.32),
            n_local_views=8,)
    elif args.mode == "simMIM":
        train_transform = MAETransform(input_size=224)
        test_transform = MAETransform(input_size=224)
    elif args.mode == "SHAM":
        anchor_transform = MAETransform(input_size=224)
        pos_transform = SimCLRTransform(input_size=224) 

    if args.mode == "simclr_supcon":
        train_dataset = CustomDataset(annotations_file=args.train_annotation, img_dir=args.img_dir, transform=TwoCropTransform(train_transform))
    else:
        if args.mode == "SHAM":
            train_dataset = CustomDataset(annotations_file=args.train_annotation, img_dir=args.img_dir, transform=anchor_transform, transform2=pos_transform, our_method=True, multi_view=args.multi_view)
        else:
            train_dataset = CustomDataset(annotations_file=args.train_annotation, img_dir=args.img_dir, transform=train_transform)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                            shuffle=True, num_workers = args.num_workers, drop_last=False)
    
    if args.mode == "simclr_supcon":
        model = SupConResNet(name=args.model, feat_dim=args.classes)
    elif args.mode == "simclr":
        model = SimCLR(model=args.model)
    elif args.mode == "mae":
        vit = vit_base_patch16_224()
        model = MAE(vit) 
    elif args.mode == "dinov2":
        model = DINOv2()
    elif args.mode == "simMIM":
        vit = torchvision.models.vit_b_16(pretrained=False)
        model = SimMIM(vit)
    elif args.mode == "SHAM":
        model = SHAM(mode=args.SHAM_mode, vit=vit_base_patch16_224())

    test_loader=None
    trainer = Trainer(model, train_loader, test_loader, args)
    trainer.train()

if __name__ == "__main__":
    args = parse_args()
    args = merge_config_with_args(args)
    set_seed(args.seed)

    main(args)
