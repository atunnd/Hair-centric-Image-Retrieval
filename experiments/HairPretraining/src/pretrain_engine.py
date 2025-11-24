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

from utils.losses import positive_consistency_loss_margin, bidirectional_margin_loss, nt_xent_1anchor_2positive, S2R2Loss

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
        self.sampling_frequency = args.sampling_frequency
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
        elif self.mode == "SHAM":
            self.criterion1 = NTXentLoss(temperature=args.temp)
            self.criterion2 = positive_consistency_loss_margin
            self.criterion3 = bidirectional_margin_loss
            self.criterion4 = nn.MSELoss()
            if self.multi_view:
                self.criterion5 = S2R2Loss(tau=0.01, k_views=5)
            else:
                self.criterion5 = S2R2Loss(tau=0.01, k_views=3)

        
        # optimizer configuration
        self.optimizer = get_optimizer(self.model, self.lr, self.weight_decay, self.beta1, self.beta2)

        # choosing backbone
        self.mode_model = args.model
        
        

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
                
                if self.negative_sampling:
                    print("🔁 Loading hard neg indices")
                    self.hard_negative_memory = torch.load(os.path.join(self.save_path, f"hard_neg_indices.pt"), weights_only=False)
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
                    if self.negative_sampling:
                        self.save_path = os.path.join(self.save_path, f"{self.mode}_{self.mode_model}_{self.args.SHAM_mode}_full_face_training_hard_negative_mining")    
                    else:
                        self.save_path = os.path.join(self.save_path, f"{self.mode}_{self.mode_model}_{self.args.SHAM_mode}_full_face_training") 
                elif args.multi_view:
                    self.save_path = os.path.join(self.save_path, f"{self.mode}_{self.mode_model}_{self.args.SHAM_mode}_multi_view_decoder_7_layers") 
                elif self.no_contrastive_loss:
                    self.save_path = os.path.join(self.save_path, f"{self.mode}_{self.mode_model}_{self.args.SHAM_mode}_no_contrastive_loss")
                else:
                    if self.negative_sampling:
                        self.save_path = os.path.join(self.save_path, f"{self.mode}_{self.mode_model}_{self.args.SHAM_mode}_hard_negative_mining")    
                    else:
                        self.save_path = os.path.join(self.save_path, f"{self.mode}_{self.mode_model}_{self.args.SHAM_mode}")    
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

            #loss.backward()
            scaler.scale(loss).backward()
            #self.optimizer.step()
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
    
    def train_one_epoch_SHAM(self, epoch=0, momentum_val=0.99, scaler=None):
        self.model.train()
        running_loss_total = 0.0
        running_loss_contrastive = 0.0
        running_loss_pos_pos = 0.0
        running_loss_margin = 0.0
        running_loss_reconstruction=0.0
        running_loss_ranking=0.0
        running_loss_local_ranking=0.0

        for batch_id, batch in enumerate(tqdm(self.train_loader, desc="Training with negative samples")):
            
            images = batch
            current_m = momentum_val 
    
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                x_anchor = images['anchor'].to(self.device)
                x_pos_1 = images['pos1'].to(self.device) 
                x_pos_2 = images['pos2'].to(self.device)
                if self.multi_view:
                    x_pos_3 = images['pos3'].to(self.device) 
                    x_pos_4 = images['pos4'].to(self.device)
                
                if self.multi_view:
                    res = self.model(img_anchor=x_anchor, img_pos1=x_pos_1, img_pos2=x_pos_2, img_pos3=x_pos_3, img_pos4=x_pos_4)
                    embedding_anchor, embedding_pos1, embedding_pos2, embedding_pos3, embedding_pos4, masked_prediction, masked_GT = res['anchor'], res['pos1'], res['pos2'], res['pos3'], res['pos4'], res['masked_prediction'], res['masked_GT']
                else:
                    res = self.model(img_anchor=x_anchor, img_pos1=x_pos_1, img_pos2=x_pos_2)
                    embedding_anchor, embedding_pos1, embedding_pos2, masked_prediction, masked_GT = res['anchor'], res['pos1'], res['pos2'], res['masked_prediction'], res['masked_GT']

                # if self.negative_sampling:
                #     if (epoch + 1) >= self.warm_up_epochs:
                #     #if (epoch+1 - self.warm_up_epochs) % self.sampling_frequency == 0:
                #         #if (epoch+1) % self.sampling_frequency == 0:
                #         if (epoch+1) - self.warm_up_epochs == 0:
                #             #K, m_star = self.estimate_K_by_PCA(embedding_anchor)
                #             if batch_id ==0:
                #                 print("=> Sampling new cluster\n")
                #                 self.hard_negative_memory = []
                #             K=6
                #             centroids, kmeans = self.run_kmeans(embedding_anchor, K, self.device_id)
                #             hard_neg_ids = self.mine_hard_negatives(embedding_anchor, centroids, kmeans)
                #             self.hard_negative_memory.append(hard_neg_ids.detach().cpu().numpy())
                #             #print(f"Estimated K = {K} (m* = {m_star})")
                #             if batch_id == len(self.train_loader) - 1:
                #                 print("=>> Save hard neg indices")
                #                 file_name = os.path.join(self.save_path, f"hard_neg_indices.pt")
                #                 torch.save(self.hard_negative_memory, file_name)
                                
                #         else:
                #             hard_neg_ids = self.hard_negative_memory[batch_id]
                #     else:
                #         hard_neg_ids = sample_random_hard_negatives(embedding_anchor)
                #         if epoch == 0:
                #             self.hard_negative_memory.append(hard_neg_ids.detach().cpu().numpy())
                #         else:
                #             self.hard_negative_memory[batch_id] = hard_neg_ids.detach().cpu().numpy()

                #     embedding_hard_negative = embedding_anchor[hard_neg_ids]
                
                if batch_id == 0:
                    with torch.no_grad():
                        emb_a = embedding_anchor
                        emb_p1_n = embedding_pos1
                        emb_a_n = F.normalize(embedding_anchor, dim=-1)
                        emb_p1_n = F.normalize(embedding_pos1, dim=-1)
                        mean_norm = emb_a.norm(dim=-1).mean().item()
                        pos_cos_mean = (emb_a_n * emb_p1_n).sum(dim=-1).mean().item()
                        max_logit = (emb_a_n @ emb_p1_n.t()).max().item()
                    print(f"[DBG] emb_norm={mean_norm:.4f}, pos_cos_mean={pos_cos_mean:.4f}, max_logit={max_logit:.4f}")

                if self.no_contrastive_loss is False:
                    contrastive_loss = self.criterion1(embedding_anchor, embedding_pos1) # contrastive loss
                reconstruction_loss = self.criterion4(masked_prediction, masked_GT) 
                if self.negative_sampling:
                    # if (epoch + 1) >= self.warm_up_epochs:
                    #     pos_consistency_loss = self.criterion2(embedding_pos1, embedding_pos2, m_p=0.3) # Positive–positive consistency loss
                    #     bidirectional_margin_loss = self.criterion3(embedding_anchor, embedding_pos1, embedding_pos2, embedding_hard_negative, m_p=0.3, m_n=0.5)
                    # else:
                    #     pos_consistency_loss = self.criterion2(embedding_pos1, embedding_pos2, m_p=0.1) # Positive–positive consistency loss
                    #     bidirectional_margin_loss = self.criterion3(embedding_anchor, embedding_pos1, embedding_pos2, embedding_hard_negative, m_p=0.1, m_n=0.7)
                    # total_loss = contrastive_loss + 0.5*reconstruction_loss + 0.2*bidirectional_margin_loss + 0.1*pos_consistency_loss
                    # ranking_loss=0
                    ranking_loss = self.criterion5(torch.cat([embedding_anchor, embedding_pos1, embedding_pos2], dim=0))
                    total_loss = 0.5*contrastive_loss + 0.3*reconstruction_loss + ranking_loss
                elif self.multi_view:
                    ranking_loss = self.criterion5(torch.cat([embedding_anchor, embedding_pos1, embedding_pos2,embedding_pos3,embedding_pos4], dim=0))
                    #total_loss = 0.2*contrastive_loss + 0.3*reconstruction_loss + ranking_loss
                    total_loss = 0.5*contrastive_loss + 0.5*reconstruction_loss + 0.5*ranking_loss
                else:
                    ranking_loss = self.criterion5(torch.cat([embedding_anchor, embedding_pos1, embedding_pos2], dim=0))
                    if self.no_contrastive_loss:
                        total_loss = 0.3*reconstruction_loss + ranking_loss
                    else:
                        total_loss = 0.5*contrastive_loss + 0.3*reconstruction_loss + 0.5*ranking_loss
            
            running_loss_total += total_loss.item()
            if self.no_contrastive_loss is False:
                running_loss_contrastive += contrastive_loss.item()
            running_loss_reconstruction += reconstruction_loss.item()
            running_loss_ranking += ranking_loss.item()

            scaler.scale(total_loss).backward()
            scaler.step(self.optimizer)
            scaler.update()
            self.optimizer.zero_grad()
            
            update_momentum(self.model.student_backbone, self.model.teacher_backbone, m=current_m)
            update_momentum(self.model.student_head, self.model.teacher_head, m=current_m)
            

            # del (
            #     x_anchor, x_pos_1, x_pos_2,
            #     embedding_anchor, 
            #     embedding_pos1, 
            #     embedding_pos2,
            #     masked_prediction, masked_GT,
            #     total_loss, 
            #     reconstruction_loss,
            #     ranking_loss
            # )
            
            # if self.no_contrastive_loss is False:
            #     del (
            #         contrastive_loss
            #     )
            
            # if self.multi_view:
            #     del (
            #         embedding_pos3, 
            #         embedding_pos4
            #     )

        with open(self.log_file, 'a') as f:
            f.write(f"\nEpoch {epoch}: Total Loss = {running_loss_total/len(self.train_loader):.6f}, Contrastive Loss = {running_loss_contrastive/len(self.train_loader):.6f}, Pos-pos Loss = {running_loss_pos_pos/len(self.train_loader):.6f}, Bi-margin Loss = {running_loss_margin/len(self.train_loader):.6f}, Reconstruction loss = {running_loss_reconstruction/len(self.train_loader):.6f}, Ranking loss = {running_loss_ranking/len(self.train_loader):.6f}\n")

        return running_loss_total/len(self.train_loader), running_loss_contrastive/len(self.train_loader), running_loss_pos_pos/len(self.train_loader), running_loss_margin/len(self.train_loader), running_loss_reconstruction/len(self.train_loader), running_loss_ranking/len(self.train_loader)
    
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
        elif self.mode == "SHAM":
            train_one_epoch = self.train_one_epoch_SHAM


        for epoch in range(self.start_epoch, self.epochs):
            print(f"Epoch {epoch}/{self.epochs}")
            if self.mode=="SHAM":
                total_loss, contrastive_loss, pos_pos_loss, bi_margin_loss, reconstruction_loss, ranking_loss = train_one_epoch(epoch=epoch, momentum_val=self.momentum_ema, scaler=self.scaler)
                print(f"Total train loss: {total_loss:.6f}, Contrastive Loss: {contrastive_loss:.6f}, Pos-pos Loss: {pos_pos_loss:.6f}, Bi-margin Loss: {bi_margin_loss:.6f}, Reconstruction Loss: {reconstruction_loss:.6f}, Ranking Loss: {ranking_loss:.6f}")
            else:
                total_loss = train_one_epoch(epoch=epoch, alpha=0, scaler=self.scaler)
                print(f"Train loss: {total_loss:.4f}")
            if (epoch+1) % 50 == 0:
                file_name = os.path.join(self.save_path, f"model_ckpt_{epoch}.pth")
                if self.mode=="SHAM":
                    self.save_checkpoint(epoch, file_name, total_loss, contrastive_loss, pos_pos_loss, bi_margin_loss, reconstruction_loss)
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
                self.save_checkpoint(epoch, file_name, total_loss, contrastive_loss, pos_pos_loss, bi_margin_loss, reconstruction_loss)
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
    
    def save_checkpoint(self, epoch, file_name, total_loss, contrastive_loss, pos_pos_loss, bi_margin_loss, reconstruction_loss):
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
            'Pos-pos loss': pos_pos_loss,
            'Bi-margin loss': bi_margin_loss,
            'Reconstruction loss': reconstruction_loss,
        }

        print("Saving checkpoint safely to CPU...")
        torch.save(checkpoint, file_name)

        import gc
        gc.collect()
        torch.cuda.empty_cache()

            