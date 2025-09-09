import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os

import lightly
from lightly.loss import NTXentLoss
from utils.utils import get_optimizer, linear_increase_alpha, margin_decay, mse_alignment_loss
from utils.transform import positive_transform, negative_transform, PositiveMaskingTransform

from .backbone import DINO
from .neg_sampling import NegSamplerClasses, NegSamplerRandomly, NegSamplerNN
import timm
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import deactivate_requires_grad, update_momentum
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F

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
        self.device_id = args.device_id

        if self.mode == 'mae':
            self.criterion = nn.MSELoss()
        elif self.mode == 'simclr':
            self.criterion = NTXentLoss()
        elif self.mode == 'simclr_supcon':
            self.criterion = SupConLoss()
        elif self.mode == "dino":
            self.criterion = DINOLoss(output_dim=2048,warmup_teacher_temp_epochs=5,)
        elif self.mode == "simMIM":
            self.criterion = nn.L1Loss()

        self.optimizer = get_optimizer(self.model, self.lr, self.weight_decay, self.beta1, self.beta2)
        self.neg_sampling = False
        self.neg_loss = args.neg_loss
        self.warm_up_epochs = self.epochs
        self.mode_model = args.model

        if args.neg_sample:

            # neg samling
            self.neg_sampling = True
            self.warm_up_epochs = args.warm_up_epochs
            self.sampling_frequency = args.sampling_frequency

            self.margin_step=0

            self.save_path = os.path.join(self.save_path, f"{self.mode}_{self.mode_model}_neg_sample_supervised_mse_static_alpha_patch_{args.atn_pooling}_{args.fusion_type}")
            print("Training with supervised neg sample")
            
            os.makedirs(self.save_path, exist_ok=True)
            print("Create save directory: ", self.save_path)

            self.log_file = os.path.join(self.save_path, 'training_log.txt')
            with open(self.log_file, 'w') as f:  # 'w' để tạo mới hoặc reset file
                f.write("Training Log - Loss per Epoch\n")

            self.log_dir = os.path.join(self.save_path, "logs")
            self.writer = SummaryWriter(log_dir=self.log_dir)  # Thư mục lưu log, tự động tạo nếu chưa có
            
            # init triplet loss
            self.triplet_loss_stage1 = nn.TripletMarginLoss(margin=0.7, p=2, eps=1e-7)
            self.triplet_loss_stage2 = nn.TripletMarginLoss(margin=0.5, p=2, eps=1e-7)

            # masking transform
            self.positive_masking_transform = PositiveMaskingTransform(mask_ratio_range=(0.1, 0.5))

            # extract negative sample
            self.negative_batch_idx =[]

            # for batch in tqdm(self.train_loader, desc="Negative mining"):
            #     #images = batch[0][1]
            #     images = batch[0][1]
            #     images = images.to(self.device)
            #     self.negative_batch_idx.append(NegSamplerNN(images, 7, 'cosine'))

        else:
            self.save_path = os.path.join(self.save_path, f"{self.mode}_{self.mode_model}")
            os.makedirs(self.save_path, exist_ok=True)
    
    def train_one_epoch_simclr(self, epoch=0, alpha=0, scaler=None):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training"):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                images = batch[0]
                x0, x1 = images[0], images[1]
                x0 = x0.to(self.device)
                x1 = x1.to(self.device)
                z0 = self.model(x0)
                z1 = self.model(x1)

                loss = self.criterion(z0, z1)
                running_loss += loss.detach()

            #loss.backward()
            scaler.scale(loss).backward()
            #self.optimizer.step()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad() 
        return running_loss / len(self.train_loader)
    
    def train_one_epoch_mae(self, epoch=0, alpha=0, scaler=None):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training"):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                views = batch[0]
                images = views[0].to(self.device)
                predictions, targets = self.model(images)
                loss = self.criterion(predictions, targets)
                running_loss += loss.detach()
            #loss.backward()
            scaler.scale(loss).backward()
            #self.optimizer.step()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

        return running_loss / len(self.train_loader)

    def train_one_epoch_simclr_supcon(self, epoch=0, alpha=0, scaler=None):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training with simclr on supcon"):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
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

            #loss.backward()
            scaler.scale(loss).backward()
            #self.optimizer.step()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

        return running_loss / len(self.train_loader)
    
    def train_one_epoch_dino(self, epoch=0, alpha=0, scaler=None):
        self.model.train()
        running_loss = 0.0
        momentum_val = cosine_schedule(epoch, self.epochs, 0.996, 1)
        for batch in tqdm(self.train_loader, desc="Training with DINO"):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                views = batch[0]
                update_momentum(self.model.student_backbone, self.model.teacher_backbone, m=momentum_val)
                update_momentum(self.model.student_head, self.model.teacher_head, m=momentum_val)
                views = [view.to(self.device) for view in views]
                global_views = views[:2]
                teacher_out = [self.model.forward_teacher(view) for view in global_views]
                student_out = [self.model.forward(view) for view in views]
                loss = self.criterion(teacher_out, student_out, epoch=epoch)
                running_loss += loss.detach()
            #loss.backward()
            scaler.scale(loss).backward()
            # We only cancel gradients of student head.
            self.model.student_head.cancel_last_layer_gradients(current_epoch=epoch)
            #self.optimizer.step()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

        return running_loss / len(self.train_loader)

    def train_one_epoch_simMIM(self, epoch=0, alpha=0, scaler=None):
        running_loss =0.0
        for batch in tqdm(self.train_loader, desc="Training with simMIM"):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                views = batch[0]
                images = views[0].to(self.device)  # views contains only a single view
                predictions, targets = self.model(images)

                loss = self.criterion(predictions, targets)
                running_loss += loss.detach()

            #loss.backward()
            scaler.scale(loss).backward()
            #self.optimizer.step()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()
        
        return running_loss/len(self.train_loader)
    
    def train_one_epoch_simclr_neg_sample(self, epoch=0, alpha=0, neg_batch_idx=None, momentum_val=0, scaler=None):
        self.model.train()
        running_loss = 0.0
        running_loss1 = 0.0
        running_loss2 = 0.0
        running_loss3 = 0.0
        running_post_dist =.0
        running_neg_dist=.0
        running_margin_violations=0
        #triplet_loss = nn.TripletMarginLoss(margin=trip_margin, p=2, eps=1e-7)

        for batch_id, batch in enumerate(tqdm(self.train_loader, desc="Training with negative samples")):
            images, hair_region_idx = batch[0], batch[1]
            current_m = momentum_val  # Hoặc: current_m = 0.996 + (0.004 * min(1, epoch / self.warm_up_epochs))
        
            update_momentum(self.model.backbone, self.model.backbone_momentum, m=current_m)
            update_momentum(self.model.projection_head, self.model.projection_head_momentum, m=current_m)
            update_momentum(self.model.decoder, self.model.decoder_momentum, m=current_m)
            trip_loss = 0.0

            if self.neg_loss == "mae":
                images0 = images[0].to(self.device)
                images1 = images0.clone()
            elif self.neg_loss == "simclr":
                images0, images1, images2 = images[0], images[1], images[2]
                images0 = images0.to(self.device)
                images1 = images1.to(self.device)
                images2 = images2.to(self.device)

                hair_region_idx0, hair_region_idx1, hair_region_idx2 = hair_region_idx[0], hair_region_idx[1], hair_region_idx[2]
                hair_region_idx0 = hair_region_idx0.to(self.device)
                hair_region_idx1 = hair_region_idx1.to(self.device)
                hair_region_idx2 = hair_region_idx2.to(self.device)

            ### STAGE 1: Randomly negative sampling
            if self.warm_up_epochs > epoch + 1:
                negative_samples = NegSamplerRandomly(images2)

            ### STAGE 2: Hard negative mining
            else:
                negative_samples = images2[self.negative_batch_idx[batch_id]]

            with torch.cuda.amp.autocast():
                # print("image0: ", images0.shape, hair_region_idx0.shape)
                # print("image1: ", images1.shape, hair_region_idx1.shape)
                # print("image2: ", images2.shape, hair_region_idx2.shape)
                anchor_batch, anchor_batch_patch = self.model(images0, hair_region_idx=hair_region_idx0)
                pos_samples = images1
                pos_batch, pos_batch_patch = self.model(pos_samples, hair_region_idx=hair_region_idx1)

                if "vit" in str(self.mode_model):
                    with torch.no_grad():
                        neg_batch, neg_batch_patch = self.model.forward_momentum(negative_samples, hair_region_idx=hair_region_idx2)
                    pos_batch_prediction, pos_batch_target = self.model(pos_samples, reconstruction=True)
                    #print("anchor: ", anchor_batch.shape) # [32, 512]
                    #print("pos sample: ", pos_batch.shape) # [32, 512]
                    #print("neg sample: ", neg_batch.shape) # [32, 512]
                    #print("masked pos batch: ", pos_batch_prediction.shape, pos_batch_target.shape) # [32, 97, 768] [32, 97, 768]
                else:
                    neg_batch, neg_batch_patch = self.model(negative_samples)
                    pos_samples = positive_transform(images1)
                    pos_batch, pos_batch_patch = self.model(pos_samples)
                    masked_pos_samples= self.positive_masking_transform(pos_samples)
                    masked_pos_batch, masked_pos_batch_patch = self.model(masked_pos_samples)
                    masked_pos_batch = masked_pos_batch.detach()

                # if masked_pos_batch_patch is not None:
                #     masked_pos_batch_patch = masked_pos_batch_patch.detach()

            if self.neg_loss == "simclr":
                neg_batch = F.normalize(neg_batch, p=2, dim=1)
                pos_batch = F.normalize(pos_batch, p=2, dim=1)
                anchor_batch = F.normalize(anchor_batch, p=2, dim=1)
                #masked_pos_batch = F.normalize(masked_pos_batch, p=2, dim=1)
                if neg_batch_patch is not None:
                    neg_batch_patch = F.normalize(neg_batch_patch, p=2, dim=1)
                    pos_batch_patch = F.normalize(pos_batch_patch, p=2, dim=1)
                    anchor_batch_patch = F.normalize(anchor_batch_patch, p=2, dim=1)

            # Debug distances
            # Debug distances
            with torch.no_grad():
                pos_dist = torch.norm(anchor_batch - pos_batch, p=2, dim=1)
                neg_dist = torch.norm(anchor_batch - neg_batch, p=2, dim=1)
                if self.warm_up_epochs > epoch + 1:
                    margin = self.triplet_loss_stage1.margin
                else:
                    margin = self.triplet_loss_stage2.margin
                #margin = trip_margin
                violations = (pos_dist - neg_dist + margin > 0)  # True = bị phạt
                running_post_dist += pos_dist.mean().item()
                running_neg_dist += neg_dist.mean().item()
                running_margin_violations += violations.sum().item()

            with torch.cuda.amp.autocast():
                if self.warm_up_epochs > epoch + 1:
                    if neg_batch_patch is None:
                        trip_loss = self.triplet_loss_stage1(anchor_batch, pos_batch, neg_batch)
                    else:
                        trip_loss = self.triplet_loss_stage1(anchor_batch_patch, pos_batch_patch, neg_batch_patch)
                    beta = 0.2
                else:
                    if neg_batch_patch is None:
                        trip_loss = self.triplet_loss_stage2(anchor_batch, pos_batch, neg_batch)
                    else:
                        trip_loss = self.triplet_loss_stage2(anchor_batch_patch, pos_batch_patch, neg_batch_patch)
                    beta = 0.2

                #trip_loss = triplet_loss(anchor_batch, pos_batch, neg_batch)
                running_loss1 += trip_loss.item()
                
                if self.neg_loss == "simclr":
                    nt_xent_loss = self.criterion(pos_batch, anchor_batch)
                    running_loss2 += nt_xent_loss.item()

                    if "vit" in str(self.mode_model):
                        mse_loss = F.mse_loss(pos_batch_prediction, pos_batch_target, reduction="mean")
                    else:
                        if neg_batch_patch is None:
                            mse_loss = F.mse_loss(pos_batch, masked_pos_batch, reduction='mean')
                        else:
                            mse_loss = F.mse_loss(pos_batch_patch, masked_pos_batch_patch, reduction="mean")
                    running_loss3 += mse_loss.item()
                    
                    total_loss = nt_xent_loss + alpha*trip_loss + beta * mse_loss
                    

                running_loss += total_loss.detach()

            #total_loss.backward()
            #self.optimizer.step()
            scaler.scale(total_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

        
        self.writer.add_scalar('Epoch/Current', epoch, global_step=epoch)
        self.writer.add_scalars(
            'Loss/Avg_per_Epoch',  # Nhóm dưới tag này để dễ xem theo epoch
            {
                'total': running_loss/len(self.train_loader),
                'nt_xent': running_loss2/len(self.train_loader),
                'triplet': running_loss1/len(self.train_loader)
            },
            global_step=epoch  # Sử dụng epoch trực tiếp làm step cho log epoch-level
        )

        if self.neg_loss == "simclr":
            with open(self.log_file, 'a') as f:
                f.write(f"Epoch {epoch}: Total Loss = {running_loss/len(self.train_loader):.4f}, NT-Xent Loss = {running_loss2/len(self.train_loader):.4f}, MSE Loss = {running_loss3/len(self.train_loader):.4f}, Triplet Loss = {running_loss1/len(self.train_loader):.4f}, Alpha = {alpha:.4f}, Pos distance: {running_post_dist/len(self.train_loader)}, Neg distance: {running_neg_dist/len(self.train_loader)}, Margin violations: {running_margin_violations/len(self.train_loader)}\n")
        elif self.neg_loss == "mae":
            with open(self.log_file, 'a') as f:
                f.write(f"Epoch {epoch}: Total Loss = {running_loss/len(self.train_loader):.4f}, MAE Loss = {running_loss2/len(self.train_loader):.4f}, Triplet Loss = {running_loss1/len(self.train_loader):.4f}, Alpha = {alpha:.4f}, Pos distance: {running_post_dist/len(self.train_loader)}, Neg distance: {running_neg_dist/len(self.train_loader)}, Margin violations: {running_margin_violations/len(self.train_loader)} \n")

        return running_loss/len(self.train_loader),running_loss1/len(self.train_loader), running_loss2/len(self.train_loader), neg_batch
    
    def train(self):
        if self.mode == "mae":
            train_one_epoch = self.train_one_epoch_mae
        elif self.mode == 'simclr':
            train_one_epoch = self.train_one_epoch_simclr
        elif self.mode == 'simclr_supcon':
            train_one_epoch = self.train_one_epoch_simclr_supcon
        elif self.mode == "dino":
            train_one_epoch = self.train_one_epoch_dino
        elif self.mode == "simMIM":
            train_one_epoch = self.train_one_epoch_simMIM

        
        scaler = torch.cuda.amp.GradScaler() 

        print(f"Training model {self.mode} with losses {self.criterion}")
        
        if self.neg_sampling:
            train_one_epoch = self.train_one_epoch_simclr_neg_sample
            neg_batch_idx= [torch.Tensor([]) for _ in range(len(self.train_loader))]
            
        for epoch in range(self.epochs):
            if self.neg_sampling:
                # if self.warm_up_epochs <= epoch + 1:
                #     #alpha = linear_increase_alpha(start_alpha=.01, current_epoch=(epoch+1)-self.warm_up_epochs, max_epochs=self.epochs-self.warm_up_epochs)
                # else:
                #     alpha = 0.01
                alpha = 0.5
            #     alpha = linear_increase_alpha(start_alpha=.01, current_epoch=(epoch+1), max_epochs=self.epochs-100)
            #     margin = margin_decay(epoch=(epoch+1),
            #                             total_epochs=self.epochs-100,
            #                             min_margin=0.5,
            #                             max_margin=0.7,
            #                             step=self.margin_step)
            print(f"Epoch {epoch}/{self.epochs}")
            if self.neg_sampling:
                momentum_val = cosine_schedule(epoch, self.epochs, 0.996, 1)
                train_loss, train_trip_loss, train_ntxent_loss, neg_batch = train_one_epoch(epoch=epoch, alpha=alpha, neg_batch_idx=neg_batch_idx, momentum_val=momentum_val, scaler=scaler)
                if self.neg_loss == "simclr":
                    print(f"Total train loss: {train_loss:.4f}, Triplet loss: {train_trip_loss}, NT-Xent loss: {train_ntxent_loss},  Alpha: {alpha}")
                elif self.neg_loss == "mae":
                    print(f"Total train loss: {train_loss:.4f}, Triplet loss: {train_trip_loss}, MAE loss: {train_ntxent_loss},  Alpha: {alpha}")
            else:
                train_loss = train_one_epoch(epoch=epoch, alpha=alpha, scaler=scaler)
                print(f"Train loss: {train_loss:.4f}")
            if (epoch+1) % 20 == 0:
                output = os.path.join(self.save_path, f"model_ckpt_{epoch}.pth")
                torch.save(self.model.state_dict(), output)
                print(f"✅ Model saved to {self.save_path}")

        self.writer.close()  # Giải phóng resource
            

