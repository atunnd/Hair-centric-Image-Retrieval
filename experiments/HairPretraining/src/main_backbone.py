import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from timm.models.vision_transformer import vit_base_patch16_224
from lightly.models import utils

import torch
import torch.nn as nn
import torch.nn.functional as F


import torch
import torch.nn as nn
import torch.nn.functional as F

from lightly.models.modules import MAEDecoderTIMM, SimCLRProjectionHead, DINOProjectionHead
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM

# class ProjectionHead(nn.Module):
#     def __init__(self, input_dim=512, hidden_dim=2048, output_dim=128, use_bn=True):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.bn1 = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
#         self.relu = nn.ReLU(inplace=True)
#         self.fc2 = nn.Linear(hidden_dim, output_dim)
#         self.bn2 = nn.BatchNorm1d(output_dim) if use_bn else nn.Identity()

#     def forward(self, x):
#         x = self.fc1(x)
#         x = self.bn1(x)
#         x = self.relu(x)
#         x = self.fc2(x)
#         x = self.bn2(x)
#         # L2 normalize for contrastive loss
#         x = F.normalize(x, dim=-1)
#         return x
from torch.nn import Module
from lightly.models.utils import (
    random_block_mask,
    update_drop_path_rate,
    update_momentum,
)
from functools import partial

def freeze_eval_module(module: Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()

class SHAM(nn.Module):
    """
    MAE-style framework with EMA teacher and dual projectors:
        - proj_global: for global-level contrastive loss (CLS or pooled)
        - proj_local:  for local-level contrastive loss (patch embeddings)
    Supports both pixel reconstruction and embedding prediction modes.
    """

    def __init__(
        self,
        vit=None,
        decoder_dim=768,
        mask_ratio=0.75,
        mode="embedding",
        pooling="mean",
        momentum=0.996,
    ):
        super().__init__()
        assert mode in ["reconstruction", "embedding"]
        assert pooling in ["mean", "attention"]

        self.mode = mode
        self.mask_ratio = mask_ratio
        self.pooling = pooling
        self.momentum = momentum

        # ========== Base ViT ==========
        if vit is None:
            vit = vit_base_patch16_224(pretrained=False)

        self.teacher_backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.teacher_backbone.sequence_length
        self.patch_size = vit.patch_embed.patch_size[0]
        embed_dim = vit.embed_dim
        self.embed_dim = embed_dim


        # ========== Decoder ==========
        self.pixel_decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=2,
            decoder_num_heads=8,
            mlp_ratio=4.0,
        )
        
        self.embedding_decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=4,
            decoder_num_heads=8,
            mlp_ratio=4.0,
        )

        # ========== Backbone ==========
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        update_drop_path_rate(
            self.student_backbone.vit,
            drop_path_rate=0.1,  # we recommend using smaller rates like 0.1 for vit-s-14
            mode="uniform",
        )
        freeze_eval_module(self.teacher_backbone)

        # ========== Projection Head ==========
        projection_head = SimCLRProjectionHead(
            input_dim=decoder_dim,
            hidden_dim=decoder_dim,
            output_dim=512
        )
        self.teacher_head = projection_head
        self.teacher_cls_head = self.teacher_patch_head = self.teacher_head
        
        self.student_head = copy.deepcopy(self.teacher_head)
        self.student_cls_head = self.student_patch_head = self.student_head
        freeze_eval_module(self.teacher_head)

    # ---------------- Encoder ----------------
    def forward_encoder_student(self, images, idx_keep=None):
        return self.student_backbone.encode(images=images, idx_keep=idx_keep)
    
    @torch.no_grad()
    def forward_encoder_teacher(self, images, idx_keep=None):
        return self.teacher_backbone.encode(images=images, idx_keep=idx_keep)

    # ---------------- Pixel Decoder ----------------
    def forward_pixel_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.pixel_decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.pixel_decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.pixel_decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.pixel_decoder.predict(x_pred)
        return x_pred

    # ---------------- Embedding Decoder ----------------
    def forward_embedding_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.embedding_decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.embedding_decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.embedding_decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.embedding_decoder.predict(x_pred)
        return x_pred

    # ---------------- Full Forward ----------------
    def forward(self, img_anchor, img_pos1, img_pos2):
        batch_size = img_anchor.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=img_anchor.device,
        )
        
        # forward anchor
        anchor_encoded = self.forward_encoder_student(images=img_anchor, idx_keep=idx_keep)
        anchor_pred = self.forward_pixel_decoder(x_encoded=anchor_encoded, idx_keep=idx_keep, idx_mask=idx_mask)
        anchor_embedding = self.forward_embedding_decoder(x_encoded=anchor_encoded, idx_keep=idx_keep, idx_mask=idx_mask)
        anchor_embedding = self.student_head(anchor_embedding.mean(dim=1))
        # get image patches for masked tokens
        patches = utils.patchify(img_anchor, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)
        
        # forward pos1
        with torch.no_grad():
            pos1_encoded = self.forward_encoder_teacher(images=img_pos1)
            pos1_embedding = self.teacher_head(pos1_encoded[:, 1:, :].mean(dim=1))
        
        # forward pos2
        with torch.no_grad():
            pos2_encoded = self.forward_encoder_teacher(images=img_pos2)
            pos2_embedding = self.teacher_head(pos2_encoded[:, 1:, :].mean(dim=1))
        
        return {
            "anchor": anchor_embedding,
            "pos1": pos1_embedding,
            "pos2": pos2_embedding,
            'masked_prediction': anchor_pred,
            'masked_GT': target
        }
        
    def extract_features(self, images):
        x_encoded = self.forward_encoder_student(images)
        return x_encoded[:, :1].mean(dim=1)


