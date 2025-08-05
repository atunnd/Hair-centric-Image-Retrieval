import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

import lightly
from lightly.loss import NTXentLoss
from utils.losses import SupConLoss
from utils.utils import get_optimizer, linear_decay_alpha
from utils.transform import positive_transform 

from .backbone import DINO
from .neg_sampling_centroid import NegSamplerCentroid
from .neg_sampling_minibatch import NegSamplerMiniBatch
import timm


class Trainer:
    def __init__(self, model, train_loader, val_loader, args):
        self.model = model.to(args.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.epochs = args.epochs
        self.device = args.device
        self.save_path = args.save_path
        self.mode = args.mode
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.beta1 = args.beta1
        self.beta2 = args.beta2

        if self.mode == 'mae':
            self.criterion = nn.MSELoss()
        elif self.mode == 'simclr':
            self.criterion = NTXentLoss()
        elif self.mode == 'simclr_supcon':
            self.criterion = SupConLoss()
        
        self.optimizer = get_optimizer(self.model, self.lr, self.weight_decay, self.beta1, self.beta2)
        self.neg_sampling = False

        if args.neg_sample:
            # neg samling
            self.neg_sampling = True
            self.neg_minibatch = args.neg_minibatch
            self.warm_up_epochs = args.warm_up_epochs
            self.centroid_momentum = args.centroid_momentum
            os.makedirs(os.path.join(self.save_path, f"{self.mode}_neg_sample_minibatch"), exist_ok=True)
            #backbone = torch.hub.load('facebookresearch/dino:main', 'dino_vits16', pretrained=True)
            #input_dim = backbone.embed_dim    
            #self.dino = DINO(backbone, input_dim)

            # Load pretrained DINO model (e.g., dino_vitb16)
            self.dino = timm.create_model("vit_base_patch16_224_dino", pretrained=True)

            # Đặt chế độ eval nếu chỉ dùng để extract features hoặc inference
            self.dino.eval()

            self.dino = self.dino.to(self.device)

            # init negative sampling
            if self.neg_minibatch:
                self.neg_sampler = NegSamplerMiniBatch(k=20)
            else:
                self.neg_sampler = NegSamplerCentroid(k=20)

            # init triplet loss
            self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        else:
            if not args.test:
                os.makedirs(os.path.join(self.save_path, self.mode), exist_ok=True)
            
        
    
    def train_one_epoch_simclr(self, epoch=0):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training"):
            x0, x1 = batch[0]
            x0, x1 = x0.to(self.device), x1.to(self.device)
            z0 = self.model(x0)
            z1 = self.model(z1)
            loss = self.criterion(z0, z1)
            running_loss += loss.detach()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad() 
        return running_loss / len(self.train_loader)
    
    def train_one_epoch_mae(self, epoch=0):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training"):
            views = batch[0]
            images = views[0].to(self.device)
            predictions, targets = self.model(images)
            loss = self.criterion(predictions, targets)
            running_loss += loss.detach()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return running_loss / len(self.train_loader)

    def train_one_epoch_simclr_supcon(self, epoch=0, alpha=0):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training"):
            images, labels = batch[0], batch[1].to(self.device)
            images = [img.to(self.device) for img in images]
            images = torch.cat([images[0], images[1]], dim=0)
            bsz = labels.shape[0]
            
            features = self.model(images)
            f1, f2 = torch.split(features, [bsz, bsz], dim=0)
            features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
            loss = self.criterion(features, labels)
            running_loss += loss.detach()
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return running_loss / len(self.train_loader)
    
    def train_one_epoch_simclr_neg_sample(self, epoch=0, alpha=0):
        self.model.train()
        running_loss = 0.0
        global_centroids = []
        for batch in tqdm(self.train_loader, desc="Training with negative samples"):
            images, labels = batch[0], batch[1].to(self.device)
            images = [img.to(self.device) for img in images]
            trip_loss = 0.0
            if self.warm_up_epochs == epoch + 1:
                # generate positive sample
                pos_batch = positive_transform(images[1])
                with torch.no_grad():
                    embeddings = self.dino()
    
                if global_centroids is None:
                    centroids, neg_batch = self.neg_sampler.fit(embeddings, first_batch=True)
                    global_centroids = centroids
                else:
                    centroids, neg_batch = self.neg_sampler.fit(embeddings, prev_centroids=global_centroids)
                trip_loss = self.triplet_loss(images[0], pos_batch, neg_batch)
            
            # simclr running
            x0, x1 = images
            x0, x1 = x0.to(self.device), x1.to(self.device)
            z0 = self.model(x0)
            z1 = self.model(x1)
            nt_xent_loss = self.criterion(z0, z1)
            
            total_loss = (1-alpha)*trip_loss + alpha*nt_xent_loss
            running_loss += total_loss.detach()

            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
        return running_loss/len(self.train_loader)

    def validate(self):
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for inputs, targets in tqdm(self.val_loader, desc="Validating"):
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                _, predictions = torch.max(outputs, dim=1) 
                correct += (predictions == targets).sum().item()
                total += targets.size(0)
            
        return correct / total
    
    def train(self):
        if self.mode == "mae":
            train_one_epoch = self.train_one_epoch_mae
        elif self.mode == 'simclr':
            train_one_epoch = self.train_one_epoch_simclr
        elif self.mode == 'simclr_supcon':
            train_one_epoch = self.train_one_epoch_simclr_supcon
        
        if self.neg_sampling:
            train_one_epoch = self.train_one_epoch_simclr_neg_sample

        for epoch in range(self.epochs):
            alpha = 0
            if self.warm_up_epochs == epoch + 1:
                alpha = linear_decay_alpha(epoch, self.epochs)
            print(f"Epoch {epoch}/{self.epochs}")
            train_loss = train_one_epoch(epoch=epoch, alpha=alpha)
            print(f"Train loss: {train_loss:.4f}, Alpha: {alpha}")
            if epoch + 1 % 20 == 0:
                val_acc = self.validate()
                print(f"Val acc: {val_acc:.4f}")
                if self.neg_sampling:
                    output = os.path.join(os.path.join(self.save_path, f"{self.mode}_neg_sample_minibatch"), f"model_ckpt_{epoch}.pth")
                else:
                    output = os.path.join(os.path.join(self.save_path, f"{self.mode}"), f"model_ckpt_{epoch}.pth")
                torch.save(self.model.state_dict(), output)
                print(f"✅ Model saved to {self.save_path}")
            

