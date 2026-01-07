import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import math

import lightly
from lightly.loss import NTXentLoss
from utils.utils import get_optimizer, linear_increase_alpha, margin_decay, mse_alignment_loss, get_latest_checkpoint, sample_random_hard_negatives
from utils.transform import positive_transform, negative_transform, PositiveMaskingTransform

from utils.losses import positive_consistency_loss_margin, bidirectional_margin_loss, nt_xent_1anchor_2positive, S2R2Loss, DistillationLoss, DenseLoss
from .neg_sampling import NegSamplerRandomly, NegSamplerStatic
#from utils.losses import DINOLoss, IBOTPatchLoss
import timm
from lightly.utils.scheduler import cosine_schedule
from lightly.models.utils import deactivate_requires_grad, update_momentum
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from lightly.loss import KoLeoLoss
from lightly.models.utils import (
    random_block_mask,
    update_drop_path_rate,
    update_momentum,
)
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule
import random

import faiss
import numpy as np
from lightly.utils.scheduler import cosine_schedule
from lightly.models import utils
from lightly.loss import MSNLoss

class Trainer:
    def __init__(self, model, train_loader, val_loader, args):

        # load model
        self.model = model.to(args.device)
        # load dataloader
        self.train_loader = train_loader
        # training setting
        self.epochs = args.epochs
        self.device = args.device
        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.beta1 = args.beta1
        self.beta2 = args.beta2
        self.device_id = args.device_id
        self.args = args
        self.save_path = args.save_path
        self.start_epoch = 0
        # choosing mode
        self.mode = args.mode
        self.momentum_ema = args.ema
        self.scaler = torch.cuda.amp.GradScaler() 
        
        # memory for hard negative
        self.negative_sampling = args.negative_sampling
        self.hard_negative_memory = []
        self.warm_up_epochs = args.warm_up_epochs
        #self.sampling_frequency = args.sampling_frequency
        self.multi_view = args.multi_view
        self.no_contrastive_loss = args.no_contrastive_loss

        ##########################################
        #    Setting loss function for each mode #
        ##########################################
        if self.mode == 'mae':
            self.criterion = nn.MSELoss()
        elif self.mode == 'simclr':
            self.criterion = NTXentLoss(temperature=args.temp)
        elif self.mode == 'simclr_supcon':
            self.criterion = SupConLoss()
        elif self.mode == "dinov2":
            #self.criterion = DINOLoss(output_dim=2048,warmup_teacher_temp_epochs=5,)
            self.criterion1 = DINOLoss()
            self.criterion2 = IBOTPatchLoss()
            self.criterion3 = KoLeoLoss()
            self.criterion = "Total loss DINO, IBOTPatchLoss, KoLeoLoss"
        elif self.mode == "simMIM":
            self.criterion = nn.L1Loss()
        elif self.mode == "DenseCL":
            self.criterion_global = NTXentLoss(memory_bank_size=(4096, 512))
            self.criterion_local = NTXentLoss(memory_bank_size=(4096, 512))
        elif self.mode == "MSN":
            self.criterion = MSNLoss()
        elif self.mode == "SHAM":
            self.criterion1 = NTXentLoss(temperature=args.temp)
            self.criterion2 = DistillationLoss()
            self.criterion3 = DenseLoss()
            self.triplet_loss_stage1 = nn.TripletMarginLoss(margin=0.7, p=2, eps=1e-7)
            self.triplet_loss_stage2 = nn.TripletMarginLoss(margin=0.5, p=2, eps=1e-7)
            self.criterion5 = S2R2Loss(tau=0.01, k_views=3)
            self.positive_masking_transform = PositiveMaskingTransform(mask_ratio_range=(0.1, 0.5))
            self.ablation = args.ablation
            if self.ablation == "fixed_margin_0_7":
                self.triplet_loss_stage1 = self.triplet_loss_stage2 = nn.TripletMarginLoss(margin=0.7, p=2, eps=1e-7)
            elif self.ablation == "fixed_margin_0_5":
                self.triplet_loss_stage1 = self.triplet_loss_stage2 = nn.TripletMarginLoss(margin=0.5, p=2, eps=1e-7)
        
        # optimizer configuration
        if not self.mode == "DenseCL":
            self.optimizer = get_optimizer(self.model, self.lr, self.weight_decay, self.beta1, self.beta2)
        else:
            params = [
                *list(model.anchor_backbone.parameters()),
                *list(model.anchor_projection_head.parameters()),
                model.prototypes,
            ]
            self.optimizer = torch.optim.Adam(params, self.lr)

        # choosing backbone
        self.mode_model = args.model

        self.negative_batch_idx =[]
        self.k = args.k
        

        ####################################
        #        Loading checkpoint        #
        ####################################
        if self.args.continue_training:
            try:
                latest_ckpt_path = get_latest_checkpoint(args.checkpoint_folder)
                checkpoint = torch.load(latest_ckpt_path, map_location=self.device, weights_only=False)
                self.save_path = args.checkpoint_folder
                print(f"✅ Found checkpoint: {latest_ckpt_path}")
            except (FileNotFoundError, TypeError):
                print("⚠️ No valid checkpoint found, starting from scratch.")
                self.start_epoch = 0
                global_loss, local_loss = 0.0, 0.0
            else:
                # Load model weights
                self.model.load_state_dict(checkpoint['model_state_dict'])

                # Load scaler
                self.scaler.load_state_dict(checkpoint['scaler_state_dict'])

                # Khởi tạo optimizer mới khớp với model hiện tại
                self.optimizer = get_optimizer(
                    self.model,
                    self.args.lr,
                    self.args.weight_decay,
                    self.args.beta1,
                    self.args.beta2
                )

                # Load optimizer state
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

                # Load epoch và các thông tin bổ sung
                self.start_epoch = checkpoint.get('epoch', 0) + 1
                global_loss = checkpoint.get('global_loss', 0.0)
                local_loss = checkpoint.get('local_loss', 0.0)

                print(f"🔁 Loaded checkpoint from epoch {self.start_epoch}")

                # Đảm bảo optimizer trên đúng device (đặc biệt khi map_location='cpu')
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
                
                # if self.negative_sampling:
                #     print("🔁 Loading hard neg indices")
                #     self.negative_batch_idx= torch.load(os.path.join(self.save_path, f"hard_neg_indices.pt"), weights_only=False)
        else:
            self.start_epoch = 0
            global_loss, local_loss = 0.0, 0.0


        ####################################
        #    Creating saving directory     #
        ####################################    
        self.momentum_ema = args.ema
        
        
        if not self.args.continue_training:
            if args.mode=="SHAM":
                if args.full_face_training:
                    self.save_path = os.path.join(self.save_path, f"{self.mode}_{self.mode_model}_full_face_training") 
                elif self.ablation != "None":
                    self.save_path = os.path.join(self.save_path, f"{self.mode}_{self.mode_model}_ablation_{self.ablation}_k_{self.k}") 
                else:
                    self.save_path = os.path.join(self.save_path, f"{self.mode}_{self.mode_model}_k_{self.k}")    
            else: 
                if args.full_face_training:
                    self.save_path = os.path.join(self.save_path, f"{self.mode}_{self.mode_model}_full_face_training")
                else:
                    self.save_path = os.path.join(self.save_path, f"{self.mode}_{self.mode_model}")
                
        if not self.args.continue_training:
            print(f"Save {args.mode} at {self.save_path}")      
            os.makedirs(self.save_path, exist_ok=True)
            new_log=True
        else:
            new_log=False


        mode = 'a' if not new_log else 'w'
        self.log_file = os.path.join(self.save_path, 'training_log.txt')
        with open(self.log_file, mode) as f: 
            if new_log:
                f.write("Training Log - Loss per Epoch\n")
            else:
                f.write("\n---- Resume training ----\n")

        self.log_dir = os.path.join(self.save_path, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.log_dir) 
    
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

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad() 
        
        with open(self.log_file, 'a') as f:
            f.write(f"\nEpoch {epoch}: Total Loss = {running_loss/len(self.train_loader):.6f}\n")
            
        return running_loss / len(self.train_loader)
    
    def train_one_epoch_msn(self, epoch=0, alpha=0, scaler=None):
        self.model.train()
        running_loss = 0.0
        for batch in tqdm(self.train_loader, desc="Training"):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                views = batch[0]
                utils.update_momentum(self.model.anchor_backbone, self.model.backbone, m=0.996)
                utils.update_momentum(
                    self.model.anchor_projection_head, self.model.projection_head, m=0.996
                )
                
                views = [view.to(self.device, non_blocking=True) for view in views]
                targets = views[0]
                anchors = views[1]
                anchors_focal = torch.concat(views[2:], dim=0)

                targets_out = self.model.backbone(targets)
                targets_out = self.model.projection_head(targets_out)
                anchors_out = self.model.forward_masked(anchors)
                anchors_focal_out = self.model.forward_masked(anchors_focal)
                anchors_out = torch.cat([anchors_out, anchors_focal_out], dim=0)

                loss = self.criterion(anchors_out, targets_out, self.model.prototypes.data)
                running_loss += loss.detach()

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad() 
        
        with open(self.log_file, 'a') as f:
            f.write(f"\nEpoch {epoch}: Total Loss = {running_loss/len(self.train_loader):.6f}\n")
            
        return running_loss / len(self.train_loader)
    
    
    def train_one_epoch_densecl(self, epoch=0, alpha=0, scaler=None):
        self.model.train()
        running_loss = 0.0
        running_global_loss =0.0
        running_local_loss = 0.0
        momentum = cosine_schedule(epoch, self.epochs, 0.996, 1)
        for batch in tqdm(self.train_loader, desc="Training"):
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                x_query, x_key = batch[0]
                utils.update_momentum(self.model.backbone, self.model.backbone_momentum, m=momentum)
                utils.update_momentum(
                    self.model.projection_head_global, self.model.projection_head_global_momentum, m=momentum
                )
                utils.update_momentum(
                    self.model.projection_head_local, self.model.projection_head_local_momentum, m=momentum
                )
                
                x_query = x_query.to(self.device)
                x_key = x_key.to(self.device)
                query_features, query_global, query_local = self.model(x_query)
                key_features, key_global, key_local = self.model.forward_momentum(x_key)

                key_local = utils.select_most_similar(query_features, key_features, key_local)
                query_local = query_local.flatten(end_dim=1)
                key_local = key_local.flatten(end_dim=1)

                loss_global = self.criterion_global(query_global, key_global)
                loss_local = self.criterion_local(query_local, key_local)
                lambda_ = 0.5
                loss = (1 - lambda_) * loss_global + lambda_ * loss_local
                running_loss += loss.detach()
                running_global_loss += loss_global.item()
                running_local_loss += loss_local.item()

            scaler.scale(loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad() 
        
        with open(self.log_file, 'a') as f:
            f.write(f"\nEpoch {epoch}: Total Loss = {running_loss/len(self.train_loader):.6f}\n")
            
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
        
        with open(self.log_file, 'a') as f:
            f.write(f"\nEpoch {epoch}: Total Loss = {running_loss/len(self.train_loader):.6f}\n")

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
    
    def train_one_epoch_dinov2(self, epoch=0, alpha=0, scaler=None):
        self.model.train()
        running_loss = 0.0
        self.total_steps = self.epochs * len(self.train_loader)
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training with dinov2")):
            views = batch[0]
            views = [view.to(self.device) for view in views]
            global_views = torch.cat(views[:2])
            local_views = torch.cat(views[2:])

            # Masking
            B = len(global_views)
            sequence_length = self.model.teacher_backbone.sequence_length
            mask = global_views.new_zeros((B, sequence_length), dtype=torch.bool)

            H, W = self.model.teacher_backbone.vit.patch_embed.grid_size
            assert (
                H * W == sequence_length - 1
            ), f"Unexpected grid size: {H}x{W}, sequence_length {sequence_length}"

            block_mask = random_block_mask(size=(B, H, W), device=mask.device)
            mask[:, 1:] = block_mask.flatten(start_dim=1)

            with torch.cuda.amp.autocast(dtype=torch.float16):
                # Teacher forward
                with torch.no_grad():
                    teacher_cls_token, teacher_features = self.model.forward_teacher(global_views)
                    teacher_cls_out = self.model.teacher_head.dino_head.forward(teacher_cls_token)
                    teacher_masked_out = self.model.teacher_head.ibot_head.forward(
                        teacher_features[mask]
                    )

                # Student forward
                student_global_cls_token, student_global_masked_features = \
                    self.model.forward_student(global_views, mask=mask)

                student_global_cls_out = self.model.student_head.dino_head.forward(
                    student_global_cls_token
                )
                student_global_masked_out = self.model.student_head.ibot_head.forward(
                    student_global_masked_features
                )
                student_local_cls_token, _ = self.model.forward_student(local_views, mask=None)
                student_local_cls_out = self.model.student_head.dino_head.forward(
                    student_local_cls_token
                )
                student_cls_out = torch.cat([student_global_cls_out, student_local_cls_out])

                # Loss
                global_step = epoch * len(self.train_loader) + batch_idx
                teacher_temp = linear_warmup_schedule(
                    step=global_step,
                    warmup_steps=int(30 / self.epochs * self.total_steps),
                    start_value=0.04,
                    end_value=0.07,
                )
                dino_loss = self.criterion1(
                    teacher_out=teacher_cls_out.chunk(2),
                    student_out=student_cls_out.chunk(len(views)),
                    teacher_temp=teacher_temp,
                )
                ibot_loss = self.criterion2(
                    teacher_out=teacher_masked_out,
                    student_out=student_global_masked_out,
                    mask=block_mask,
                    teacher_temp=teacher_temp,
                )
                koleo_loss = 0.1 * sum(
                    self.criterion3(t) for t in student_global_cls_token.chunk(2)
                )
                loss = dino_loss + ibot_loss + koleo_loss

            running_loss += loss.detach()

            # ✅ Mixed Precision update
            scaler.scale(loss).backward()

            # zero lr for last layer if needed
            if epoch < 1:
                for param_group in self.optimizer.param_groups:
                    if "last_layer" in param_group:
                        param_group["lr"] = 0.0

            # weight decay schedule
            weight_decay = cosine_schedule(
                step=global_step,
                max_steps=self.total_steps,
                start_value=0.04,
                end_value=0.4,
            )
            for group in self.optimizer.param_groups:
                if group["weight_decay"] != 0.0:
                    group["weight_decay"] = weight_decay

            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()

            # momentum update teacher
            momentum = cosine_schedule(
                step=global_step,
                max_steps=self.total_steps,
                start_value=0.992,
                end_value=1.0,
            )
            update_momentum(self.model.student_backbone, self.model.teacher_backbone, m=momentum)
            update_momentum(self.model.student_head, self.model.teacher_head, m=momentum)

        avg_loss = running_loss / len(self.train_loader)
        print(f"epoch: {epoch:>02}, loss: {avg_loss:.5f}")
        return avg_loss

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
    
    # -----------------------------
    # Step 1. Estimate K by PCA (SCAN-style)
    # -----------------------------
    def estimate_K_by_PCA(self, X: torch.Tensor, explained_var_threshold=0.9, scale_factor=2.0, max_K=2000):
        """Ước lượng số cụm K bằng PCA cumulative variance ratio"""
        X_np = X.detach().cpu().numpy().astype('float32')
        N, D = X_np.shape
        n_components = min(D, N - 1)
        pca = faiss.PCAMatrix(D, n_components)
        pca.train(X_np)
        eigenvalues = faiss.vector_to_array(pca.eigenvalues).astype('float32')
        explained_ratio = eigenvalues / np.sum(eigenvalues)
        cumulative_ratio = np.cumsum(explained_ratio)
        m_star = np.searchsorted(cumulative_ratio, explained_var_threshold) + 1
        K_est = int(np.clip(scale_factor * m_star, 5, min(max_K, N - 1)))
        return K_est, m_star


    # -----------------------------
    # Step 2. K-Means clustering
    # -----------------------------
    def run_kmeans(self, X: torch.Tensor, K: int, device_id: int = 0):
        """Chạy K-means bằng FAISS, trả về centroids tensor""" 
        X_np = X.detach().cpu().numpy().astype('float32') 
        D = X_np.shape[1] 
        use_gpu = faiss.get_num_gpus() > 0 
        kmeans = faiss.Kmeans(d=D, k=K, niter=20, gpu=False, verbose=True) 
        kmeans.train(X_np) 
        centroids = torch.from_numpy(kmeans.centroids).to(X.device) 
        return centroids, kmeans

    # -----------------------------
    # Step 3. Hard negative mining (cluster-based)
    # -----------------------------
    def mine_hard_negatives(self, anchor: torch.Tensor, centroids: torch.Tensor, kmeans):
        """
        Chọn hard negatives cho từng anchor bằng cách:
        - tìm 2 centroids gần nhất
        - dùng centroid thứ 2 (neighbor) để tìm các mẫu gần nó nhất
        """
        device = anchor.device
        N, D = anchor.shape

        # --- 1. Tìm 2 centroid gần nhất ---
        D_c, I_c = kmeans.index.search(anchor.detach().cpu().numpy().astype('float32'), 2)
        neighbor_centroid_ids = torch.from_numpy(I_c[:, 1]).long().to(device)  # (N,)

        # --- 2. Build FAISS index từ các sample ---
        index = faiss.IndexFlatL2(D)
        index.add(anchor.detach().cpu().numpy().astype('float32'))

        # --- 3. Tìm top-k sample gần từng centroid ---
        topk = 5
        D_samp, I_samp = index.search(centroids.detach().cpu().numpy().astype('float32'), topk)
        I_samp = torch.from_numpy(I_samp).long().to(device)  # (K, topk)

        # --- 4. Chọn hard negative cho từng anchor ---
        rand_offsets = torch.randint(0, topk, (N,), device=device)
        hard_neg_ids = I_samp[neighbor_centroid_ids, rand_offsets]

        # Tránh chọn chính anchor → thay bằng phần tử thứ 0 nếu trùng
        same_mask = hard_neg_ids == torch.arange(N, device=device)
        hard_neg_ids[same_mask] = I_samp[neighbor_centroid_ids[same_mask], 0]

        # --- 5. Lấy embeddings ---
        #hard_negatives = anchor[hard_neg_ids]
        return hard_neg_ids
    
    def train_one_epoch_SHAM(self, epoch=0, momentum_val=0.99, scaler=None, prev_margin_violations=0):
        self.model.train()
        running_loss_total = 0.0
        running_loss_contrastive = 0.0
        running_loss_triplet =0.0
        running_loss_mse =0.0
        running_loss_dense=0.0
        running_loss_ranking=0.0
        running_loss_distillation=0.0
        running_post_dist = 0.0
        running_neg_dist = 0.0
        running_margin_violations = 0.0
        total_k=0.0
        momentum = cosine_schedule(epoch, self.epochs, 0.996, 1)
        
        for batch_id, batch in enumerate(tqdm(self.train_loader, desc="Training with negative samples")):
            self.optimizer.zero_grad()
            
            # update backbone momentum
            # update header
            #update_momentum(self.model.student_backbone, self.model.teacher_backbone, m=momentum)
            #update_momentum(self.model.student_head, self.model.teacher_head, m=momentum)
            #update_momentum(self.model.student_fusion_head, self.model.teacher_fusion_head, m=momentum)
            current_m = momentum_val  
            update_momentum(self.model.backbone, self.model.backbone_momentum, m=current_m)
            update_momentum(self.model.projection_head, self.model.projection_head_momentum, m=current_m)
            
            images = batch
            current_m = momentum_val 
    
            x_anchor = images['anchor'].to(self.device)
            x_pos_1 = images['pos1'].to(self.device) 
            #x_pos_2 = images['pos2'].to(self.device)
            # if self.multi_view:
            #     x_pos_3 = images['pos3'].to(self.device) 
            
            if self.ablation == "None" or self.ablation != "randomly" or  self.ablation != "fixed_hard":
                if self.warm_up_epochs > epoch + 1:     #STAGE 1: RANDOMLY NEGATIVE MINING
                    negative_samples = NegSamplerRandomly(x_pos_1)
                else:
                    if (epoch + 1) == self.warm_up_epochs:
                        # if batch_id == 0:
                        #     self.negative_batch_idx = []

                        #     B = x_anchor.shape[0]
                        #     v = prev_margin_violations/B
                        #     x = max(2, math.floor((1 - v) * 10))
                        #     y = x + 5
                        #     random_k = random.randint(x, y)
                        #     total_k = random_k
                        #     print(f"\n=>[x, y] = [{x}, {y}]\n")
                        total_k = random_k = self.k
                            
                        self.negative_batch_idx.append(NegSamplerStatic(self.model, x_pos_1, k=total_k)) # negative with momentum model
                    
                        if batch_id == len(self.train_loader) - 1:
                            print("==> Hard neg indices saved!")
                            file_name = os.path.join(self.save_path, f"hard_neg_indices.pt")
                            torch.save(self.negative_batch_idx, file_name)

                    # print("Len idx: ", len(self.negative_batch_idx[batch_id]))
                    negative_samples = x_pos_1[self.negative_batch_idx[batch_id]]
                    # print("Negative samples: ", negative_samples[0].min(), negative_samples[0].max())
                    
            elif self.ablation == "randomly":
                negative_samples = NegSamplerRandomly(x_pos_1)
                
            elif self.ablation == "fixed_hard":
                if (epoch + 1) == self.warm_up_epochs:
                    if batch_id == 0:
                        self.negative_batch_idx = []

                        B = x_anchor.shape[0]
                        v = prev_margin_violations/B
                        x = max(2, math.floor((1 - v) * 10))
                        y = x + 5
                        random_k = random.randint(x, y)
                        total_k = random_k
                        print(f"\n=>[x, y] = [{x}, {y}]\n")
                        
                    self.negative_batch_idx.append(NegSamplerStatic(self.model, x_pos_1, k=total_k)) # negative with momentum model
                
                    if batch_id == len(self.train_loader) - 1:
                        print("==> Hard neg indices saved!")
                        file_name = os.path.join(self.save_path, f"hard_neg_indices.pt")
                        torch.save(self.negative_batch_idx, file_name)
                negative_samples = x_pos_1[self.negative_batch_idx[batch_id]]
        
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                # res = self.model(x_anchor, x_pos_2, x_pos_3)
                # anchor_batch_s, anchor_batch_t, pos_batch, pos_batch_2 = res["anchor_s"], res["anchor_t"], res["pos_contrastive"], res["pos2_contrastive"]
                # anchor_ranking, pos_batch_ranking, pos_batch_2_ranking = res["anchor_ranking"], res["pos_ranking"], res["pos2_ranking"]
                # anchor_patch, pos_patch = res["anchor_patch"], res["pos_patch"]
                neg_batch = self.model(negative_samples)
                pos_samples = positive_transform(x_pos_1)
                pos_batch = self.model(pos_samples)
                anchor_batch = self.model(x_anchor)
                masked_pos_samples = self.positive_masking_transform(pos_samples)
                masked_pos_batch = self.model.forward_momentum(masked_pos_samples)
                
            
            neg_batch = F.normalize(neg_batch, p=2, dim=1)
            pos_batch = F.normalize(pos_batch, p=2, dim=1)
            anchor_batch = F.normalize(anchor_batch, p=2, dim=1)
            masked_pos_batch = F.normalize(masked_pos_batch, p=2, dim=1)
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

            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                # Forward triplet loss
                if self.ablation != "No_Triplet":
                    if self.warm_up_epochs > epoch + 1:
                        triplet_loss = self.triplet_loss_stage1(anchor_batch, pos_batch, neg_batch)
                    else:
                        triplet_loss = self.triplet_loss_stage2(anchor_batch, pos_batch, neg_batch)
                    running_loss_triplet += triplet_loss.item()
                
                # Forward contrastive loss
                contrastive_loss = self.criterion1(pos_batch, anchor_batch)
                running_loss_contrastive += contrastive_loss.item()
                
                # Forward MSE losss
                if self.ablation != "No_MSE":
                    mse_loss = F.mse_loss(pos_batch, masked_pos_batch, reduction='mean')
                    running_loss_mse += mse_loss.item()
                
                
                # Total loss
                if self.ablation == "No_Triplet":
                    total_loss = contrastive_loss + 0.2*mse_loss
                elif self.ablation == "No_MSE":
                    total_loss = contrastive_loss + 0.5*triplet_loss
                else:
                    total_loss = contrastive_loss + 0.5*triplet_loss + 0.2*mse_loss
                
                
                running_loss_total += total_loss.item()

            scaler.scale(total_loss).backward()
            scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            scaler.step(self.optimizer)
            scaler.update()

            
        with open(self.log_file, 'a') as f:
            f.write(f"\nEpoch {epoch}: Total Loss = {running_loss_total/len(self.train_loader):.6f}, Contrastive Loss = {running_loss_contrastive/len(self.train_loader):.6f}, Triplet Loss = {running_loss_triplet/len(self.train_loader):.6f}, MSE loss = {running_loss_mse/len(self.train_loader):.6f}, Positive distance = {running_post_dist/len(self.train_loader):.6f}, Negative distance = {running_neg_dist/len(self.train_loader):.6f}, Margin violations: {running_margin_violations/len(self.train_loader)}, Total k: {total_k} \n")
        
        return running_loss_total/len(self.train_loader), running_loss_contrastive/len(self.train_loader), running_loss_triplet/len(self.train_loader), running_loss_mse/len(self.train_loader), prev_margin_violations, running_post_dist/len(self.train_loader), running_neg_dist/len(self.train_loader), running_margin_violations/len(self.train_loader)
    
    def train(self):
        if self.mode == "mae":
            train_one_epoch = self.train_one_epoch_mae
        elif self.mode == 'simclr':
            train_one_epoch = self.train_one_epoch_simclr
        elif self.mode == 'simclr_supcon':
            train_one_epoch = self.train_one_epoch_simclr_supcon
        elif self.mode == "dinov2":
            train_one_epoch = self.train_one_epoch_dinov2
        elif self.mode == "simMIM":
            train_one_epoch = self.train_one_epoch_simMIM
        elif self.mode == "DenseCL":
            train_one_epoch = self.train_one_epoch_densecl
        elif self.mode == "MSN":
            train_one_epoch = self.train_one_epoch_msn
        elif self.mode == "SHAM":
            train_one_epoch = self.train_one_epoch_SHAM


        prev_margin_violations = 0
        for epoch in range(self.start_epoch, self.epochs):
            print(f"Epoch {epoch}/{self.epochs}")
            if self.mode=="SHAM":
                total_loss, contrastive_loss, triplet_loss, mse_loss, prev_margin_violations, pos_dis, neg_dis, margin_violation = train_one_epoch(epoch=epoch, momentum_val=self.momentum_ema, scaler=self.scaler, prev_margin_violations=prev_margin_violations)
                print(f"Total train loss: {total_loss:.6f}, Contrastive Loss: {contrastive_loss:.6f}, Triplet Loss: {triplet_loss:.6f}, MSE Loss: {mse_loss:.6f}, Pos distance: {pos_dis:.6f}, Neg distance: {neg_dis:.6f}, Margin violations: {margin_violation}")
            else:
                total_loss = train_one_epoch(epoch=epoch, alpha=0, scaler=self.scaler)
                print(f"Train loss: {total_loss:.4f}")
            if (epoch+1) % 50 == 0:
                file_name = os.path.join(self.save_path, f"model_ckpt_{epoch}.pth")
                if self.mode=="SHAM":
                    self.save_checkpoint(epoch, file_name, total_loss, contrastive_loss, triplet_loss, mse_loss)
                else:
                    checkpoint = {
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'scaler_state_dict': self.scaler.state_dict(),
                        'args': self.args,
                        'Total_loss': total_loss,
                    }
                    torch.save(checkpoint, file_name)
                    print(f"✅ Saved checkpoint at epoch {epoch} -> {file_name}")
                    
            file_name = os.path.join(self.save_path, f"model_ckpt_latest.pth")
            if self.mode=="SHAM":
                self.save_checkpoint(epoch, file_name, total_loss, contrastive_loss, triplet_loss, mse_loss )
            else:
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scaler_state_dict': self.scaler.state_dict(),
                    'args': self.args,
                    'Total_loss': total_loss,
                    }
                torch.save(checkpoint, file_name)
                print(f"✅ Saved checkpoint at epoch {epoch} -> {file_name}")

        self.writer.close()  # Giải phóng resource
    
    def save_checkpoint(self, epoch, file_name, total_loss, contrastive_loss, triplet_loss, mse_loss):
        # Move model and optimizer to CPU before saving
        model_cpu = {k: v.detach().cpu() for k, v in self.model.state_dict().items()}
        optimizer_cpu = {
            'state': {
                k: {kk: vv.detach().cpu() if torch.is_tensor(vv) else vv
                    for kk, vv in v.items()}
                for k, v in self.optimizer.state_dict()['state'].items()
            },
            'param_groups': self.optimizer.state_dict()['param_groups']
        }
        scaler_cpu = self.scaler.state_dict()

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_cpu,
            'optimizer_state_dict': optimizer_cpu,
            'scaler_state_dict': scaler_cpu,
            'args': self.args,
            'Total_loss': total_loss,
            'Contrastive Loss': contrastive_loss,
            'Triplet loss': triplet_loss,
            'MSE loss': mse_loss,
        }

        print("Saving checkpoint safely to CPU...")
        torch.save(checkpoint, file_name)

        import gc
        gc.collect()
        torch.cuda.empty_cache()

            