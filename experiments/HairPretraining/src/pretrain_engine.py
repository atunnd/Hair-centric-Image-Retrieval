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
from .neg_sampling import NegSamplerMiniBatch, NegSamplerClasses, NegSamplerRandomly
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
        self.neg_loss = args.neg_loss
        self.warm_up_epochs = self.epochs

        if args.neg_sample:
            # neg samling
            self.neg_sampling = True
            self.negative_centroid = args.negative_centroid
            self.neg_minibatch = args.neg_minibatch
            self.warm_up_epochs = args.warm_up_epochs
            self.centroid_momentum = args.centroid_momentum
            self.sampling_frequency = args.sampling_frequency

            if self.negative_centroid:
                self.save_path = os.path.join(self.save_path, f"{self.mode}_neg_sample_centroid")
            else:
                self.save_path = os.path.join(self.save_path, f"{self.mode}_neg_sample")

            # set sampling method and loss
            if self.neg_loss == "simclr":
                self.neg_sampler = NegSamplerMiniBatch(k=10, dim=128, momentum=self.centroid_momentum, negative_centroid=self.negative_centroid, save_path=self.save_path)
                self.criterion = NTXentLoss()
            elif self.neg_loss == "supcon":
                self.neg_sampler = NegSamplerClasses()
                self.criterion = SupConLoss()
            
            # init triplet loss
            self.triplet_loss = nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        else:
            self.save_path = os.path.join(self.save_path, self.mode)
        os.makedirs(self.save_path, exist_ok=True)
    
    def train_one_epoch_simclr(self, epoch=0, alpha=0):
        self.model.train()
        running_loss = 0.0
        for batch, _ in tqdm(self.train_loader, desc="Training"):
            images = batch[0]
            x0, x1 = images[0], images[1]
            x0 = x0.to(self.device)
            x1 = x1.to(self.device)
            z0 = self.model(x0)
            z1 = self.model(x1)
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
        for batch in tqdm(self.train_loader, desc="Training with simclr on supcon"):
            images, labels = batch[0], batch[1]
            images = [img.to(self.device) for img in images]
            labels = labels.to(self.device)
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

    def train_one_epoch_simclr_neg_supervised(self, epoch=0, alpha=0):
        self.model.train()
        running_loss =0.0
        for batch in tqdm(self.train_loader, desc="Training with supervised negative sampling"):
            images, labels = batch[0], batch[1].to(self.device)
            images = [img.to(self.device) for img in images]

    
    def train_one_epoch_simclr_neg_sample(self, epoch=0, alpha=0, embeddings=0, init_centroid=False, global_centroids=0, ema_loss1 = 0.0, ema_loss2=0.0, beta_loss=0.99, neg_batch=None):
        self.model.train()
        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0
        for batch_id, batch in enumerate(tqdm(self.train_loader, desc="Training with negative samples")):
            images, labels = batch[0], batch[1].to(self.device)
            images = [img.to(self.device) for img in images]
            trip_loss = 0.0

            ### STAGE 1: Randomly negative sampling
            neg_batch[batch_id] = positive_transform(NegSamplerRandomly(images[0]))

            if self.warm_up_epochs == epoch + 1:
                if batch_id == 0:
                    print("create embeddings")
                pos_batch = positive_transform(images[1])
                with torch.no_grad():
                    embeddings[batch_id] = self.dino(pos_batch)

            ### STAGE 2: Hard negative mining
            if self.warm_up_epochs <= epoch:
                # generate positive sample
                if (epoch + 1) % self.sampling_frequency == 0:
                    neg_batch[batch_id] = self.neg_sampler(images[0])
                    if batch_id == 0:
                        print("Init centroids")
                    neg_batch[batch_id] = self.neg_sampler.forward(ema_embeddings, images[0], first_batch=True)

            pos_batch = positive_transform(images[1])
            trip_loss = self.triplet_loss(images[0], pos_batch, neg_batch[batch_id])
            running_loss1 += trip_loss.item()
                
            
            # Main encoder running
            x0, x1 = images
            x0, x1 = x0.to(self.device), x1.to(self.device)
            z0 = self.model(x0)
            z1 = self.model(x1)
            nt_xent_loss = self.criterion(z0, z1)
            running_loss2 += nt_xent_loss.item()
       
            # Total loss
            total_loss = nt_xent_loss + alpha*trip_loss
            running_loss += total_loss.detach()

            total_loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        return running_loss/len(self.train_loader),running_loss1/len(self.train_loader), running_loss2/len(self.train_loader), global_centroids, embeddings, ema_loss1, ema_loss2, neg_batch

    
    def train(self):
        if self.mode == "mae":
            train_one_epoch = self.train_one_epoch_mae
        elif self.mode == 'simclr':
            train_one_epoch = self.train_one_epoch_simclr
        elif self.mode == 'simclr_supcon':
            train_one_epoch = self.train_one_epoch_simclr_supcon

        print(f"Training model {self.mode} with losses {self.criterion}")
        
        if self.neg_sampling:
            train_one_epoch = self.train_one_epoch_simclr_neg_sample
            global_centroids = [torch.Tensor([]) for _ in range(len(self.train_loader))]
            init_centroid = True
            embeddings = [torch.Tensor([]) for _ in range(len(self.train_loader))]
            neg_batch= [torch.Tensor([]) for _ in range(len(self.train_loader))]
            ema_loss1 = 0.0
            ema_loss2 = 0.0
            beta_loss = 0.99

        for epoch in range(self.epochs):
            alpha = 1
            if self.warm_up_epochs <= epoch:
                alpha = linear_decay_alpha(epoch-self.warm_up_epochs, self.epochs-self.warm_up_epochs)
            print(f"Epoch {epoch}/{self.epochs}")
            if self.neg_sampling:
                if self.warm_up_epochs == epoch:
                    train_loss, train_trip_loss, train_ntxent_loss, global_centroids, embeddings, ema_loss1, ema_loss2, neg_batch = train_one_epoch(epoch=epoch, alpha=alpha, embeddings=embeddings, init_centroid=True, global_centroids=global_centroids, ema_loss1=ema_loss1, ema_loss2=ema_loss2, beta_loss=beta_loss, neg_batch=neg_batch)
                else:
                    train_loss, train_trip_loss, train_ntxent_loss, _, _, ema_loss1, ema_loss2, _ = train_one_epoch(epoch=epoch, alpha=alpha, embeddings=embeddings, init_centroid=False, global_centroids=global_centroids, ema_loss1=ema_loss1, ema_loss2=ema_loss2, beta_loss=beta_loss, neg_batch=neg_batch)
                print(f"Total train loss: {train_loss:.4f}, Triplet loss: {train_trip_loss}, NT-Xent loss: {train_ntxent_loss},  Alpha: {alpha}")
            else:
                train_loss = train_one_epoch(epoch=epoch, alpha=alpha)
                print(f"Train loss: {train_loss:.4f}")
            if (epoch+1) % 20 == 0:
                output = os.path.join(self.save_path, f"model_ckpt_{epoch}.pth")
                torch.save(self.model.state_dict(), output)
                print(f"✅ Model saved to {self.save_path}")
            

