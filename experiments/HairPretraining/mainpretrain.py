import argparse
from src.pretrain_engine import Trainer
from utils.utils import set_seed
import yaml
import os
from utils.transform import get_test_transform, get_train_transform, TwoCropTransform
from utils.dataloader import CustomDataset
import torch
from torch.utils.data import DataLoader
from src.backbone import SupConResNet

def parse_args():
    parser = argparse.ArgumentParser(description="Self-supervised/Supervised Trainer Arguments")

    # Training config
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use: cuda or cpu')
    parser.add_argument('--save_path', type=str, default='output_dir', help='Path to save model checkpoint')
    parser.add_argument('--size', type=int, default=224, help="Image size for training")
    parser.add_argument('--train_annotation', type=str, help='Path to training annotation file')
    parser.add_argument('--test_annotation', type=str, help='Path to testing annotation file')
    parser.add_argument('--img_dir', type=str, help='Path to image directory')

    # optimization config
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default = 1e-4, help='Weight decay rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='1st moment')
    parser.add_argument('--beta2', type=float, default=0.999, help='2nd moment')

    # loss config
    parser.add_argument('--temp', type=float, default=0.7, help="temperature for loss function")

    # Model option
    parser.add_argument('--mode', type=str, default='simclr_supcon', choices=['mae', 'simclr', 'simclr_supcon'])
    parser.add_argument('--model', type=str, default='resnet18')

    # Optional config
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--config', type=str, help='Optional path to YAML config file (overrides args)')
    parser.add_argument('--num_workers', type=int, default=4)

    # testing
    parser.add_argument('--test', default=False, type=bool, help='Run in test mode')
    parser.add_argument('--test_model_path', type=str, default=None, help='Path to the model checkpoint for testing')

    # negative sampling
    parser.add_argument('--neg_sample', default=False, type=bool, help='Use negative sampling')
    parser.add_argument('--warm_up_epochs', default=0, type=int, help='Number of warmup epochs for negative sampling')
    parser.add_argument('--dino_checkpoint', type=str, help="Path to pretrained dino checkpoint")
    parser.add_argument('--centroid_momentum', type=float, default=0.9)
    parser.add_argument('--neg_minibatch', type=bool, default=False)

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
    
    # construct data loader
    mean = (0.5071, 0.4867, 0.4408) # cifar100
    std = (0.2675, 0.2565, 0.2761)
    train_transform = get_train_transform(args.size, mean, std)
    test_transform = get_test_transform(args.size, mean, std)

    train_dataset = CustomDataset(args.train_annotation, args.img_dir, TwoCropTransform(train_transform))
    test_dataset = CustomDataset(args.test_annotation, args.img_dir, test_transform)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True, num_workers = args.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, 
                             shuffle=True, num_workers = args.num_workers)
    

    model = SupConResNet(name=args.model)
    trainer = Trainer(model, train_loader, test_loader, args)
    
    if args.test:
        state_dict = torch.load(args.test_model_path, map_location=args.device)
        model.load_state_dict(state_dict)
        print(f"✅ Model loaded from {args.test_model_path}")
        val_acc = trainer.validate()
        print(f"Test accuracy: {val_acc:.4f}")
    else:
        trainer.train()

if __name__ == "__main__":
    args = parse_args()
    args = merge_config_with_args(args)
    set_seed(args.seed)

    main(args)
