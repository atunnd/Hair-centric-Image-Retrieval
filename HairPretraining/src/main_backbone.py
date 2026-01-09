import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from timm.models.vision_transformer import vit_base_patch16_224
from lightly.models import utils

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import torch
import torch.nn as nn
import torch.nn.functional as F

from lightly.models.modules import MAEDecoderTIMM, SimCLRProjectionHead, DINOProjectionHead
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM


from torch.nn import Module
from lightly.models.utils import (
    random_block_mask,
    update_drop_path_rate,
    update_momentum,
)
from functools import partial
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ViTWrapper: trả về feature map (B, C, H, W) tương thích với CNN-style heads
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class ViTWrapper(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        vit = torchvision.models.vit_b_16(weights=weights)
        vit.heads = nn.Identity()

        self.conv_proj = vit.conv_proj
        self.encoder = vit.encoder

        # ---- cls token ----
        if hasattr(vit, "class_token"):
            self.cls_token = vit.class_token
        elif hasattr(vit, "cls_token"):
            self.cls_token = vit.cls_token
        else:
            raise RuntimeError("ViT model has no class token")

        # ---- positional embedding ----
        if hasattr(vit, "pos_embedding"):
            self.pos_embedding = vit.pos_embedding
        elif hasattr(vit.encoder, "pos_embedding"):
            self.pos_embedding = vit.encoder.pos_embedding
        else:
            raise RuntimeError("ViT model has no positional embedding")

        self.patch_size = 16


    def _resize_pos_embed(self, pos_embed, num_patches, device, dtype):
        """
        Resize pretrained pos_embed to match new num_patches (if needed),
        using bilinear interpolation on the patch-grid.
        pos_embed: (1, L, D) where L = 1 + old_num_patches
        num_patches: target number of patch tokens (without cls)
        """
        if pos_embed is None:
            return None
        L = pos_embed.shape[1]
        if L == num_patches + 1:
            return pos_embed.to(device=device, dtype=dtype)

        # separate cls and patch embeddings
        cls_pe = pos_embed[:, :1, :].to(device=device, dtype=dtype)      # (1,1,D)
        patch_pe = pos_embed[:, 1:, :].to(device=device, dtype=dtype)    # (1, P_old, D)
        D = patch_pe.shape[-1]

        old_num = patch_pe.shape[1]
        old_size = int(math.sqrt(old_num))
        new_size = int(math.sqrt(num_patches))
        if old_size * old_size != old_num or new_size * new_size != num_patches:
            raise RuntimeError(f"PosEmbed reshape error: old_num={old_num}, new_num={num_patches}")

        patch_pe = patch_pe.reshape(1, old_size, old_size, D).permute(0, 3, 1, 2)  # (1,D,old_h,old_w)
        patch_pe = F.interpolate(patch_pe, size=(new_size, new_size), mode="bilinear", align_corners=False)
        patch_pe = patch_pe.permute(0, 2, 3, 1).reshape(1, new_size*new_size, D)   # (1, new_num, D)
        new_pos = torch.cat([cls_pe, patch_pe], dim=1)  # (1, 1+new_num, D)
        return new_pos

    def forward(self, x):
        n = x.shape[0]
        device = x.device
        dtype = x.dtype

        # optional: normalize input if using pretrained weights (or let caller do it)
        # if hasattr(self.encoder, '_process_input'): x = self.encoder._process_input(x)

        # Patch embeddings: conv_proj -> (n, C, H_patch, W_patch)
        x = self.conv_proj(x)                 # (n, 768, H/16, W/16)
        B, C, Hf, Wf = x.shape
        num_patches = Hf * Wf
        x = x.flatten(2).transpose(1, 2)      # (n, num_patches, C)

        # CLS token
        cls_tokens = self.cls_token.expand(n, -1, -1).to(device=device, dtype=dtype)  # (n,1,C)
        # If you prefer contiguous copy: cls_tokens = cls_tokens.clone()

        x = torch.cat((cls_tokens, x), dim=1)  # (n, num_patches+1, C)

        # Positional embeddings: resize if needed
        pos_embed = self._resize_pos_embed(self.pos_embedding, num_patches, device, dtype)
        x = x + pos_embed[:, : x.size(1)]

        # Encoder: ensure encoder expects (B, L, D)
        x = self.encoder(x)  # (n, num_patches+1, C)

        cls_token = x[:, 0]            # (n, C)
        patch_tokens = x[:, 1:, :]     # (n, num_patches, C)
        pooled_patches = patch_tokens.mean(dim=1)  # (n, C)

        # optional: return grid feature map too
        feature_map = patch_tokens.permute(0,2,1).reshape(n, C, Hf, Wf)

        return cls_token, feature_map



class RankingHeadMLP(nn.Module):
    """
    Light MLP ranking head.
    - No BatchNorm
    - Very shallow
    """

    def __init__(self, in_dim, hidden_dim=512, out_dim=256, normalize=True):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.normalize = normalize

    def forward(self, x):
        x = F.relu(self.fc1(x))
        z = self.fc2(x)

        if self.normalize:
            z = F.normalize(z, dim=1)

        return z


def freeze_eval_module(module: Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()

from torch.nn import MultiheadAttention

# ----------- Shape Head -----------
class ShapeHead(nn.Module):
    def __init__(self, in_ch=2048, out_dim=256, pooling="avg"):
        super().__init__()
        self.pooling = pooling
        self.fc = nn.Linear(in_ch, out_dim)

    def forward(self, feat): # B, C, H, W
        if self.pooling == "avg":
            x = feat.mean(dim=[2, 3])
        elif self.pooling == "max":
            x = feat.max(dim=[2, 3])
        z = self.fc(x)
        z = F.normalize(z, dim=1)
        return z

# ----------- Shape Head -----------
class TextureHead(nn.Module):
    def __init__(self, in_ch, token_dim=256, grid=(4, 4)):
        super().__init__()
        self.grid = grid
        self.token_dim = token_dim
        self.conv_proj = nn.Conv2d(in_ch, token_dim, kernel_size=1)

    def forward(self, feat): # feat: (B, C, H, W)
        # project channels
        x = self.conv_proj(feat)
        B, D, H, W = x.shape
        gh, gw = self.grid
        # adaptive pool to grid size
        x = F.adaptive_avg_pool2d(x, output_size=(gh, gw)) # B, D, gh, gw
        tokens = x.view(B, D, gh*gw).permute(0, 2, 1) # B, N, D
        tokens = F.normalize(tokens, dim=2)
        return tokens

# ----------- Cross-Attention Fusion -----------
class ShapeTextureFusion(nn.Module):
    def __init__(self, dim=256, num_heads=4, attn_dropout=0.0):
        super().__init__()
        self.attn = MultiheadAttention(embed_dim=dim, num_heads=num_heads,
                                       dropout=attn_dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim)
        )

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, shape_vec, texture_tokens):

        # make query shape token sequence length=1
        q = shape_vec.unsqueeze(1)
        # cross-atnn: query=q, key=texture, value=texture
        attn_out, attn_weights = self.attn(query=q, key=texture_tokens, value=texture_tokens)
        # attn_out: (B, 1, D)
        attn_out = attn_out.squeeze(1) # (B, D)
        # residual + FFN
        out = self.norm1(attn_out + shape_vec)
        out2 = self.ff(out)
        fused = self.norm2(out + out2)
        fused = F.normalize(fused, dim=1)
        return fused, attn_weights # fused: (B, D)

# ----------- Fusion Head ----------
class FusedHead(nn.Module):
    def __init__(self, in_ch=2048, token_dim=512, grid=(4,4), num_heads=4, out_dim=512):
        super().__init__()
        self.shape_head = ShapeHead(in_ch, out_dim=token_dim)
        self.texture_head = TextureHead(in_ch, token_dim=token_dim, grid=grid)
        self.fusion = ShapeTextureFusion(dim=token_dim, num_heads=num_heads)

        if token_dim != out_dim:
            self.out_proj = nn.Linear(token_dim, out_dim)
        else:
            self.out_proj = None

    def forward(self, feat):
        shape = self.shape_head(feat) # (B, D)
        texture_tokens = self.texture_head(feat) # (B, N, D)
        fused, attn = self.fusion(shape, texture_tokens) # (B, D)
        if self.out_proj is not None:
            fused = self.out_proj(fused)
            fused = F.normalize(fused, dim=1)
        return fused, attn

class SHAM(nn.Module):
    """
    MAE-style framework with EMA teacher and dual projectors:
        - proj_global: for global-level contrastive loss (CLS or pooled)
        - proj_local:  for local-level contrastive loss (patch embeddings)
    Supports both pixel reconstruction and embedding prediction modes.
    """

    def __init__(
        self,
        model="resnet50",
    ):
        super().__init__()

        self.model = model
        print("Using backbone:", self.model)

        if model == "resnet18":
            backbone = torchvision.models.resnet18(weights=None)
            self.teacher_backbone = nn.Sequential(*list(backbone.children())[:-2])
            proj_input_dim, output_dim = 512, 128

        elif model == "resnet50":
            backbone = torchvision.models.resnet50(weights=None)
            self.teacher_backbone = nn.Sequential(*list(backbone.children())[:-2])
            proj_input_dim, output_dim = 2048, 1024

        elif model == "vit_b_16":
            vit = ViTWrapper(weights=None)
            self.teacher_backbone = vit
            proj_input_dim, output_dim = 768, 512

        else:
            raise ValueError(f"Unsupported model: {model}")

         # ========== Ranking Head ============
        self.ranking_head = RankingHeadMLP(in_dim=proj_input_dim, out_dim=512)

        # ========= Global Head ============
        self.pool_head = nn.AdaptiveAvgPool2d(1)

        # ========== Backbone ==========
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        freeze_eval_module(self.teacher_backbone)

        # ========== Projection Head ==========
        projection_head = SimCLRProjectionHead(
            input_dim=proj_input_dim,
            hidden_dim=proj_input_dim,
            output_dim=512
        )
        self.teacher_head = projection_head
        self.teacher_cls_head = self.teacher_patch_head = self.teacher_head

        self.student_head = copy.deepcopy(self.teacher_head)
        self.student_cls_head = self.student_patch_head = self.student_head
        freeze_eval_module(self.teacher_head)

    # ---------------- Encoder ----------------
    def forward_encoder_student(self, x):
        if self.model == "vit_b_16":
            cls, feature_map = self.student_backbone(x)
        else:
            feature_map = self.student_backbone(x)
            cls = None
        return cls, feature_map

    @torch.no_grad()
    def forward_encoder_teacher(self, x):
        if self.model == "vit_b_16":
            cls, feature_map = self.teacher_backbone(x)
        else:
            feature_map = self.teacher_backbone(x)
            cls = None
        return cls, feature_map

    def feat_to_tokens(self, feat, is_vit=False):
        """
        feat:
          - CNN: (B, C, H, W)
          - ViT: (B, 1+N, D)
        return:
          tokens: (B, N, D)
        """
        if feat.dim() == 4: # CNN
            B, C, H, W = feat.shape
            tokens = feat.flatten(2).permute(0, 2, 1) # (B, H*W, C)
        else: # ViT
            tokens = feat[:, 1:, :] # drop cls (B, N, D)

        tokens = F.normalize(tokens, dim=-1)
        return tokens

    def dense_correspondence(self, anchor_tokens, pos_tokens):
        """
        anchor_tokens: (B, Na, D)
        pos_tokens:    (B, Np, D)

        return:
          idx: (B, Na)
        """
        # cosine similarity via dot product
        sim = torch.einsum("bnd, bmd->bnm", anchor_tokens, pos_tokens)

        # best matching positive patch per anchor batch
        idx = sim.argmax(dim=-1) # (B, Na)

        return idx, sim

    def gather_pos_tokens(self, pos_tokens, idx):
        """
        pos_tokens: (B, Np, D)
        idx:        (B, Na)

        return:
          pos_matched: (B, Na, D)
        """
        B, na = idx.shape
        idx_expanded = idx.unsqueeze(-1).expand(-1, -1, pos_tokens.size(-1))
        pos_matched = torch.gather(pos_tokens, dim=1, index=idx_expanded)
        return pos_matched

    def sample_patches(self, tokens, K):
        """
        tokens: (B, N, D)
        return:
          sampled_tokens: (B, K, D)
          idx: (B, K)
        """
        B, N, _ = tokens.shape
        idx = torch.randint(0, N, (B, K), device=tokens.device)
        idx_exp = idx.unsqueeze(-1).expand(-1, -1, tokens.size(-1))
        sampled = torch.gather(tokens, 1, idx_exp)
        return sampled, idx

    def forward_patch_sampling(self, anchor, positive1, positive2, K=32):
        """
        anchor:    (B, Na, D)
        positive1: (B, N1, D)
        positive2: (B, N2, D)

        return:
          A_s: (B, K, D)
          P_s: (B, K, D)   # fused positives
        """

        anchor = F.normalize(anchor, dim=-1)
        positive1 = F.normalize(positive1, dim=-1)
        positive2 = F.normalize(positive2, dim=-1)

        # -------- correspondence pos1 --------
        idx1, sim1 = self.dense_correspondence(anchor, positive1)
        P1_match = self.gather_pos_tokens(positive1, idx1)  # (B, Na, D)

        # -------- correspondence pos2 --------
        idx2, sim2 = self.dense_correspondence(anchor, positive2)
        P2_match = self.gather_pos_tokens(positive2, idx2)  # (B, Na, D)

        # -------- similarity scores --------
        # sim: (B, Na, Np) → take best match score
        w1 = torch.gather(sim1, 2, idx1.unsqueeze(-1)).squeeze(-1)  # (B, Na)
        w2 = torch.gather(sim2, 2, idx2.unsqueeze(-1)).squeeze(-1)  # (B, Na)

        w1 = w1.unsqueeze(-1)  # (B, Na, 1)
        w2 = w2.unsqueeze(-1)  # (B, Na, 1)

        # -------- fuse positives (anchor-centric) --------
        P_fused = (w1 * P1_match + w2 * P2_match) / (w1 + w2 + 1e-6)
        P_fused = F.normalize(P_fused, dim=-1)  # (B, Na, D)

        # -------- sample K patches --------
        A_s, sidx = self.sample_patches(anchor, K)
        P_s, _ = self.sample_patches(P_fused, K)

        return A_s, P_s


    def forward(self, x_anchor, x_pos_2, x_pos_3):
        # forward anchor_s and anchor_T
        anchor_s_cls, anchor_s_feat = self.forward_encoder_student(x_anchor)
        if self.model == "vit_b_16":
            anchor_s_ranking = self.ranking_head(anchor_s_cls)
            anchor_s_contrastive = self.student_head(anchor_s_cls)
        else:
            anchor_s_ranking = self.ranking_head(self.pool_head(anchor_s_feat).flatten(start_dim=1))
            anchor_s_contrastive = self.student_head(self.pool_head(anchor_s_feat).flatten(start_dim=1))
        with torch.no_grad():
            anchor_t_cls, anchor_t_feat = self.forward_encoder_teacher(x_anchor)
            if self.model == "vit_b_16":        
                anchor_t = self.teacher_head(anchor_t_cls)
            else:
                anchor_t = self.teacher_head(self.pool_head(anchor_t_feat).flatten(start_dim=1))

        # forward first pos
        pos_cls, pos_feat = self.forward_encoder_student(x_pos_2)
        if self.model == "vit_b_16":
            pos_ranking = self.ranking_head(pos_cls)
            pos_contrastive = self.student_head(pos_cls)
        else:
            pos_ranking = self.ranking_head(self.pool_head(pos_feat).flatten(start_dim=1))
            pos_contrastive = self.student_head(self.pool_head(pos_feat).flatten(start_dim=1))

        # forward second pos
        pos2_cls, pos2_feat = self.forward_encoder_student(x_pos_3)
        if self.model == "vit_b_16":
            pos2_ranking = self.ranking_head(pos2_cls)
            pos2_contrastive = self.student_head(pos2_cls)
        else:  
            pos2_ranking = self.ranking_head(self.pool_head(pos2_feat).flatten(start_dim=1))
            pos2_contrastive = self.student_head(self.pool_head(pos2_feat).flatten(start_dim=1))

        # forward patch sampling
        anchor_s_patch, pos_patch = self.forward_patch_sampling(self.feat_to_tokens(anchor_s_feat),
                                                                self.feat_to_tokens(pos_feat),
                                                                self.feat_to_tokens(pos2_feat))

        #return self.forward_encoder_student(x)
        return {
            "anchor_s": anchor_s_contrastive,
            "anchor_t": anchor_t,
            "pos_contrastive": pos_contrastive,
            "pos2_contrastive": pos2_contrastive,
            "anchor_ranking": anchor_s_ranking,
            "pos_ranking": pos_ranking,
            "pos2_ranking": pos2_ranking,
            "anchor_patch": anchor_s_patch,
            "pos_patch": pos_patch,
        }

    def extract_features(self, x):
        x_encoded = self.student_backbone(x)
        return x_encoded.mean(dim=(2,3))

    @torch.no_grad()
    def extract_features_ema(self, x):
        x_encoded = self.teacher_backbone(x)
        return x_encoded.mean(dim=(2,3))

def freeze_eval_module(module: Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()
    
class ViTWrapper(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        vit = torchvision.models.vit_b_16(weights=weights)
        vit.heads = nn.Identity()  # bỏ classifier head
        
        self.conv_proj = vit.conv_proj
        self.encoder = vit.encoder
        self.cls_token = vit.class_token
        self.pos_embedding = vit.encoder.pos_embedding

    def forward(self, x):
        n = x.shape[0]

        # Patch embeddings
        x = self.conv_proj(x)                # (n, 768, H/16, W/16)
        x = x.flatten(2).transpose(1, 2)     # (n, num_patches, 768)

        # Thêm CLS token
        cls_tokens = self.cls_token.expand(n, -1, -1)  # (n, 1, 768)
        x = torch.cat((cls_tokens, x), dim=1)          # (n, num_patches+1, 768)

        # Thêm positional embeddings
        x = x + self.pos_embedding[:, : x.size(1)]

        # Encoder
        x = self.encoder(x)  # (n, num_patches+1, 768)

        # Tách ra
        cls_token = x[:, 0]       # (n, 768)
        patch_tokens = x[:, 1:]   # (n, num_patches, 768)

        # Pooling patch tokens (trung bình)
        pooled_patches = patch_tokens.mean(dim=1)  # (n, 768)

        return cls_token, pooled_patches

class SHAM2(nn.Module):
    def __init__(self, model="resnet18"):
        super().__init__()
        self.model = model
        print("Using backbone:", self.model)

        if model == "resnet18":
            backbone = torchvision.models.resnet18(weights=None)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            proj_input_dim, output_dim = 512, 128

        elif model == "resnet50":
            backbone = torchvision.models.resnet50(weights=None)
            self.backbone = nn.Sequential(*list(backbone.children())[:-1])
            proj_input_dim, output_dim = 2048, 1024

        elif model == "vit_b_16":
            vit = ViTWrapper(weights=None)
            self.backbone = vit
            proj_input_dim, output_dim = 768, 512

        else:
            raise ValueError(f"Unsupported model: {model}")

        self.projection_head = SimCLRProjectionHead(proj_input_dim, proj_input_dim, output_dim)

        # momentum encoder
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)
        #freeze_eval_module(self.backbone_momentum)
        f#reeze_eval_module(self.projection_head_momentum)

    def forward(self, x):
        if "vit" in self.model:
            # option A: only CLS token
            cls_token, pooled_patch = self.backbone(x)
            z = self.projection_head(cls_token)
            return z

        else:  # ResNet
            x = self.backbone(x)                # [batch, feat, 1, 1]
            x = x.flatten(start_dim=1)          # [batch, feat]
            z = self.projection_head(x)
            return z

    @torch.no_grad()
    def forward_momentum(self, x):
        if "vit" in self.model:
            cls_token, pooled_patch = self.backbone_momentum(x)   
            z = self.projection_head_momentum(cls_token)
            return z
        else:
            x = self.backbone_momentum(x)
            x = x.flatten(start_dim=1)
            z = self.projection_head_momentum(x)
            return z

    def extract_features(self, x):
        if "vit" in self.model:
            cls_token, _ = self.backbone(x)
            return cls_token
        else:
            return self.backbone(x).flatten(start_dim=1)
    
    @torch.no_grad()
    def extract_features_ema(self, x):
        if "vit" in self.model:
            cls_token, _ = self.backbone_momentum(x)
            return cls_token
        else:
            return self.backbone_momentum(x).flatten(start_dim=1)
    