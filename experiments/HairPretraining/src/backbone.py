import torch
from timm.models.vision_transformer import vit_base_patch32_224
from torch import nn
import torch.nn as nn
import torch.nn.functional as F

from lightly.models import utils 
from lightly.models.modules import MAEDecoderTIMM, SimCLRProjectionHead, DINOProjectionHead
from lightly.transforms.simclr_transform import SimCLRTransform
from lightly.loss import DINOLoss
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule

import copy
from lightly.models.modules import MAEDecoderTIMM
from lightly.transforms import MAETransform
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.models.modules.masked_vision_transformer_torchvision import (
    MaskedVisionTransformerTorchvision,
)

import torch.nn as nn

import torch
import torch.nn as nn
import torchvision.models as models
from .masked_vision_transformer_timm import MaskedVisionTransformerTIMM

# Giả định SimCLRProjectionHead (thay bằng lightly nếu dùng)
# class SimCLRProjectionHead(nn.Module):
#     def __init__(self, input_dim, hidden_dim, output_dim):
#         super().__init__()
#         self.fc1 = nn.Linear(input_dim, hidden_dim)
#         self.relu = nn.ReLU()
#         self.fc2 = nn.Linear(hidden_dim, output_dim)

#     def forward(self, x):
#         x = self.relu(self.fc1(x))
#         return self.fc2(x)

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Giả sử SimCLRProjectionHead (nếu dùng lightly, import; đây là placeholder)
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Giả sử SimCLRProjectionHead (nếu dùng lightly, import; đây là placeholder)
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Giả sử SimCLRProjectionHead (nếu dùng lightly, import; đây là placeholder)

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

# Giả sử SimCLRProjectionHead (nếu dùng lightly, import; đây là placeholder)


class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=8, batch_first=True)

    def forward(self, x):  # x: [batch, seq_len, dim]
        # Use CLS as query, patches as key/value
        query = x[:, :1, :]  # [batch, 1, dim]
        key_value = x[:, 1:, :]  # [batch, num_patches, dim]
        attn_output, attn_weights = self.attn(query, key_value, key_value, need_weights=True)
        # attn_weights: [batch, 1, num_patches]
        return attn_output.squeeze(1), attn_weights.squeeze(1)  # [batch, dim], [batch, num_patches]

# Custom ViT backbone with SVF integrated
class ViTBackbone(nn.Module):
    def __init__(self, vit_model, k=100):  # k for top-k in SVF
        super().__init__()
        self.conv_proj = vit_model.conv_proj
        self.class_token = vit_model.class_token
        self.pos_embed = vit_model.encoder.pos_embedding
        self.dropout = vit_model.encoder.dropout
        self.encoder_layers = vit_model.encoder.layers  # List of transformer layers
        self.final_ln = vit_model.encoder.ln  # Final LayerNorm
        self.num_layers = len(self.encoder_layers)
        self.k = k  # Top-k for SVF
        self.hidden_dim = vit_model.hidden_dim
        self.token_importance_gen = nn.Linear(self.hidden_dim, 1)  # Ω for Z in SVF

    def forward(self, x):
        # Patch embedding
        x = self.conv_proj(x)
        x = x.flatten(2).transpose(1, 2)  # [batch, num_patches, embed_dim]
        
        # Add CLS token
        cls_token = self.class_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Dropout
        x = self.dropout(x)
        
        # Apply encoder layers up to penultimate
        for i in range(self.num_layers - 1):
            x = self.encoder_layers[i](x)
        
        # SVF at penultimate layer
        penultimate_layer = self.encoder_layers[-2]
        # To get attn_weights, call self-attn (assuming standard ViT layer with self_attention)
        self_attn = penultimate_layer.self_attention  # Adjust if layer structure differs
        attn_output, attn_weights = self_attn(query=x, key=x, value=x, need_weights=True, average_attn_weights=False)  # attn_weights [batch, num_heads, seq_len, seq_len]
        
        # Extract CLS attention to patches: A [batch, num_heads, num_patches]
        A = attn_weights[:, :, 0, 1:]  # CLS query to patches
        
        # Aggregate heads: Â = sum over heads
        A_hat = A.sum(dim=1)  # [batch, num_patches]
        
        # Token importance Z = sigmoid(Ω(E_i)), E_i are patches from x[:, 1:, :]
        patches = x[:, 1:, :]  # [batch, num_patches, dim]
        Z = torch.sigmoid(self.token_importance_gen(patches)).squeeze(-1)  # [batch, num_patches]
        
        # O = Â ⊕ (Â ⊗ Z)
        O = A_hat + (A_hat * Z)  # [batch, num_patches]
        
        # Top-k indices based on O
        _, top_indices = O.topk(self.k, dim=1, sorted=False)  # [batch, k]
        
        top_indices = top_indices.long()  # Ensure dtype for gather
        
        # Gather top-k patches
        top_patches = torch.gather(patches, 1, 
                                  top_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim))  # [batch, k, 768]
        
        # Gather positional embeddings for top-k patches
        cls_pos = self.pos_embed[:, :1, :]  # [1, 1, 768]
        # Fix: Expand pos_embed để match batch size trước khi gather
        pos_embed_expanded = self.pos_embed.expand(x.shape[0], -1, -1)[:, 1:, :]  # [batch, 196, 768]
        top_pos = torch.gather(pos_embed_expanded, 1, 
                              top_indices.unsqueeze(-1).expand(-1, -1, self.hidden_dim))  # [batch, k, 768]
        input_last_layer_pos = torch.cat([cls_pos.expand(x.shape[0], -1, -1), top_pos], dim=1)  # [batch, 1+k, 768]
        
        # New sequence for last layer: CLS + top-k patches
        cls_token = x[:, 0, :].unsqueeze(1)  # [batch, 1, 768]
        input_last_layer = torch.cat([cls_token, top_patches], dim=1)  # [batch, 1+k, 768]
        input_last_layer = input_last_layer + input_last_layer_pos  # Add pos_embed
        
        # Apply last layer
        x = self.encoder_layers[-1](input_last_layer)  # [batch, 1+k, 768]
        
        # Final LN
        x = self.final_ln(x)
        
        return x, top_patches  # [batch, 1+k, 768]

class SimCLR(nn.Module):
    def __init__(self, backbone, model=None, attention_pooling=False, k=50, fusion_type='transformer'):
        super().__init__()
        self.model = model
        self.attn_pooling = attention_pooling
        self.k = k  # For SVF
        self.fusion_type = fusion_type  # 'transformer' as requested
        
        # Handle backbone: For ViT, wrap it with SVF
        if "vit" in str(self.model):
            vit = models.vit_b_16()  # Load full ViT if backbone is not already wrapped
            self.backbone = ViTBackbone(vit, k=self.k)
            proj_input_dim = 768  # For vit_b_16
            output_dim = 512
            # Fusion module: Transformer as requested
            self.fusion_module = nn.TransformerEncoderLayer(d_model=proj_input_dim, nhead=8, batch_first=True)
        else:
            self.backbone = backbone  # Assume ResNet-like, already Sequential[:-1]
            if model == "resnet18":
                proj_input_dim = 512
                output_dim=128
            elif model == "resnet50":
                proj_input_dim = 2048
                output_dim=1024
            else:
                raise ValueError("Unsupported model")
        
        self.projection_head = SimCLRProjectionHead(proj_input_dim, proj_input_dim, output_dim)
        
        if self.attn_pooling and "vit" in str(self.model):
            self.pooled = AttentionPooling(proj_input_dim)
        elif self.attn_pooling:
            print("Warning: Attention pooling only for ViT")
        
        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        for param in self.backbone_momentum.parameters():
            param.requires_grad = False
        for param in self.projection_head_momentum.parameters():
            param.requires_grad = False

    def _internal_forward(self, backbone, projection_head, x):
        x = backbone(x)  # [batch, 1+k, dim] after SVF
        
        if "vit" in str(self.model):
            cls_token = x[:, 0, :]  # [batch, dim]
            patch_token = x[:, 1:, :]  # Top-k patches [batch, k, dim]
            
            # Fusion with transformer (as requested)
            fused_input = torch.stack([cls_token, patch_token.mean(dim=1)], dim=1)  # [batch, 2, dim] (mean patch as summary for fusion)
            fused_output = self.fusion_module(fused_input)  # [batch, 2, dim]
            final_cls = fused_output[:, 0, :]  # [batch, dim]
            
            z = projection_head(final_cls)
            return z, patch_token
            
        else:
            x = x.flatten(start_dim=1)  # For CNN like ResNet [batch, features]
            z = projection_head(x)
            return z, None

    def forward(self, x):
        return self._internal_forward(self.backbone, self.projection_head, x)

    def forward_momentum(self, x):
        return self._internal_forward(self.backbone_momentum, self.projection_head_momentum, x)
    
    def extract_features(self, x):
        features = self.backbone(x)  # [batch, 1+k, dim] for ViT
        
        if "vit" in str(self.model):
            cls_token = features[:, 0, :]
            patch_token = features[:, 1:, :]
            fused_input = torch.stack([cls_token, patch_token.mean(dim=1)], dim=1)
            fused_output = self.fusion_module(fused_input)
            final_cls = fused_output[:, 0, :]
            return final_cls  # Fused CLS
        else:
            return features.flatten(start_dim=1)


import torch
import torch.nn as nn
import copy

class SimCLR_Our(nn.Module):
    def __init__(self, vit, mask=None, max_patches=100):
        super().__init__()
        decoder_dim = 512
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_embed.patch_size[0]

        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length

        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=1,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )

        proj_input_dim = 768  # For vit_b_16
        output_dim = 512
        self.projection_head = SimCLRProjectionHead(proj_input_dim, proj_input_dim, output_dim)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)
        self.decoder_momentum = copy.deepcopy(self.decoder)
        for param in self.backbone_momentum.parameters():
            param.requires_grad = False
        for param in self.projection_head_momentum.parameters():
            param.requires_grad = False
        for param in self.decoder_momentum.parameters():
            param.requires_grad = False

    def forward_encoder(self, images, idx_keep=None, hair_region_idx=None):
        if hair_region_idx is not None:
            return self.backbone.encode(images=images, idx_keep=None, hair_region_idx=None)
        else:
            return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_encoder_momentum(self, images, idx_keep=None, hair_region_idx=None):
        if hair_region_idx is not None:
            return self.backbone_momentum.encode(images=images, idx_keep=None, hair_region_idx=None)
        else:
            return self.backbone_momentum.encode(images=images, idx_keep=idx_keep)

    def forward(self, images, reconstruction=False, hair_region_idx=None, extract_features=True):
        if reconstruction:
            batch_size = images.shape[0]
            idx_keep, idx_mask = utils.random_token_mask(
                size=(batch_size, self.sequence_length),
                mask_ratio=self.mask_ratio,
                device=images.device,
            )
            x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
            #print(f"x_encoded reconstruction: {x_encoded.shape}\n") #torch.Size([32, 50, 768])
            x_pred = self.forward_decoder(
                x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
            )
            patches = utils.patchify(images, self.patch_size)
            target = utils.get_at_index(patches, idx_mask - 1)
            x_projection = self.projection_head_momentum(x_encoded[:, 0, :])
            return x_projection, x_pred, target
        else:
            x_encoded = self.forward_encoder(images=images, idx_keep=None, hair_region_idx=None)
            x_projection = self.projection_head(x_encoded[:, 0, :])  # Use CLS token
            #print(f"x_encoded hair: {x_encoded.shape}\n")
            return x_projection, None, None

        # batch_size = images.shape[0]
        # idx_keep, idx_mask = utils.random_token_mask(
        #     size=(batch_size, self.sequence_length),
        #     mask_ratio=self.mask_ratio,
        #     device=images.device,
        # )
        # x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        # #print(f"x_encoded reconstruction: {x_encoded.shape}\n") #torch.Size([32, 50, 768])
        # x_pred = self.forward_decoder(
        #     x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        # )
        # patches = utils.patchify(images, self.patch_size)
        # target = utils.get_at_index(patches, idx_mask - 1)

        # projected_cls_token = self.projection_head(x_encoded[:, 0, :])  # Use CLS token
        # return projected_cls_token, x_pred, target

    def forward_momentum(self, images, reconstruction=False, hair_region_idx=None, extract_features=True):
        # batch_size = images.shape[0]
        # idx_keep, idx_mask = utils.random_token_mask(
        #     size=(batch_size, self.sequence_length),
        #     mask_ratio=self.mask_ratio,
        #     device=images.device,
        # )
        # x_encoded = self.forward_encoder_momentum(images=images, idx_keep=idx_keep)
        # x_pred = self.forward_decoder_momentum(
        #     x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        # )
        # patches = utils.patchify(images, self.patch_size)
        # target = utils.get_at_index(patches, idx_mask - 1)

        # projected_cls_token = self.projection_head_momentum(x_encoded[:, 0, :])  # Use CLS token
        # return projected_cls_token, x_pred, target
        if reconstruction:
            batch_size = images.shape[0]
            idx_keep, idx_mask = utils.random_token_mask(
                size=(batch_size, self.sequence_length),
                mask_ratio=self.mask_ratio,
                device=images.device,
            )
            x_encoded = self.forward_encoder_momentum(images=images, idx_keep=idx_keep)
            #print(f"x_encoded hair momentum: {x_encoded.shape}\n") #torch.Size([32, 50, 768])
            x_pred = self.forward_decoder_momentum(
                x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
            )
            patches = utils.patchify(images, self.patch_size)
            target = utils.get_at_index(patches, idx_mask - 1)

            x_projection = self.projection_head_momentum(x_encoded[:, 0, :])
            return x_projection, x_pred, target
        else:
            x_encoded = self.forward_encoder_momentum(images=images, idx_keep=None, hair_region_idx=None)
            x_projection = self.projection_head_momentum(x_encoded[:, 0, :])  # Use CLS token
            #print(f"x_encoded hair momentum: {x_encoded.shape}\n") #torch.Size([32, 50, 768])
            return x_projection, None, None
        

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)

model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
}

class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x

class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet18', head='mlp', feat_dim=128):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        feat = F.normalize(self.head(feat), dim=1)
        return feat

class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)

class DINO(torch.nn.Module):
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z

class MAE(nn.Module):
    def __init__(self, vit):
        super().__init__()

        decoder_dim = 512
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_embed.patch_size[0]

        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=1,
            decoder_num_heads=16,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
        )

    def forward_encoder(self, images, idx_keep=None):
        z = self.backbone.encode(images=images, idx_keep=idx_keep)
        return z

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(x_masked, idx_keep, x_decode.type_as(x_masked))

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def forward(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)
        return x_pred, target
    
    def extract_features(self, images):
        x_encoded = self.forward_encoder(images=images, idx_keep=None)
        return x_encoded[:, 0, :]

class DINO(torch.nn.Module):
    def __init__(self, backbone, input_dim):
        super().__init__()
        self.student_backbone = backbone
        self.student_head = DINOProjectionHead(
            input_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_backbone = copy.deepcopy(backbone)
        self.teacher_head = DINOProjectionHead(input_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_backbone)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, x):
        y = self.student_backbone(x).flatten(start_dim=1)
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.teacher_backbone(x).flatten(start_dim=1)
        z = self.teacher_head(y)
        return z
    
class SimMIM(nn.Module):
    def __init__(self, vit):
        super().__init__()

        decoder_dim = vit.hidden_dim
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_size
        self.sequence_length = vit.seq_length

        self.backbone = MaskedVisionTransformerTorchvision(vit=vit)

        # the decoder is a simple linear layer
        self.decoder = nn.Linear(decoder_dim, vit.patch_size**2 * 3)

    def forward_encoder(self, images, batch_size, idx_mask):
        # pass all the tokens to the encoder, both masked and non masked ones
        return self.backbone.encode(images=images, idx_mask=idx_mask)

    def forward_decoder(self, x_encoded):
        return self.decoder(x_encoded)
    
    def extract_features(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio = self.mask_ratio,
            device=images.device
        )
        x_encoded = self.forward_encoder(images, batch_size, idx_mask)
        return x_encoded[:, 0, :]

    def forward(self, images):
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )

        # Encoding...
        x_encoded = self.forward_encoder(images, batch_size, idx_mask)
        x_encoded_masked = utils.get_at_index(x_encoded, idx_mask)

        # Decoding...
        x_out = self.forward_decoder(x_encoded_masked)

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)

        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        return x_out, target
    

# class OriginSimCLR(nn.Module):
#     def __init__(self, backbone, model=None, attention_pooling=False):
#         super().__init__()
#         self.model = model
#         self.attn_pooling = attention_pooling
        
#         # Handle backbone: For ViT, wrap it
#         if "vit" in str(model):
#             vit = models.vit_b_16()  # Load full ViT if backbone is not already wrapped
#             self.backbone = ViTBackbone(vit)
#             proj_input_dim = 768  # For vit_b_16
#             output_dim = 512
#         else:
#             self.backbone = backbone  # Assume ResNet-like, already Sequential[:-1]
#             if model == "resnet18":
#                 proj_input_dim = 512
#                 output_dim=128
#             elif model == "resnet50":
#                 proj_input_dim = 2048
#                 output_dim=1024
#             else:
#                 raise ValueError("Unsupported model")
        
#         self.projection_head = SimCLRProjectionHead(proj_input_dim, proj_input_dim, output_dim)
        

#     def forward(self, x):
#          # ResNet: [batch, features, 1, 1]; ViT: [batch, seq_len, dim]
        
#         if "vit" in str(self.model):
#             x, top_patches = self.backbone(x) 
#             cls_token = x[:, 0, :]  # CLS token [batch, dim]
#             z = self.projection_head(cls_token)
#             return z, top_patches
#         else:
#             x = self.backbone(x) 
#             x = x.flatten(start_dim=1)  # For CNN like ResNet [batch, features]
#             z = self.projection_head(x)
#             return z
    
#     def extract_features(self, x):
#         features = self.backbone(x)  # ResNet: [batch, features, 1, 1]; ViT: [batch, seq_len, dim]
        
#         if "vit" in str(self.model):
#             embedding = features[:, 0, :]
#         else:
#             embedding = features.flatten(start_dim=1)  # [batch, features] for ResNet-like
        
#         return embedding  # Always [batch, dim] raw from backbone

class OriginSimCLR(nn.Module):
    def __init__(self, backbone, model=None):
        super().__init__()
        self.model = model
        
        # Handle backbone: For ViT, wrap it
        if "vit" in str(model):
            vit = models.vit_b_16()  # Load full ViT if backbone is not already wrapped
            self.backbone = ViTBackbone(vit)
            proj_input_dim = 768  # For vit_b_16
            output_dim = 512
        else:
            self.backbone = backbone  # Assume ResNet-like, already Sequential[:-1]
            if model == "resnet18":
                proj_input_dim = 512
                output_dim=128
            elif model == "resnet50":
                proj_input_dim = 2048
                output_dim=1024
            else:
                raise ValueError("Unsupported model")
        
        self.projection_head = SimCLRProjectionHead(proj_input_dim, proj_input_dim, output_dim)

    def forward(self, x):
        x = self.backbone(x)  # ResNet: [batch, features, 1, 1]; ViT: [batch, seq_len, dim]
        
        if "vit" in str(self.model):
            if self.attn_pooling:
                cls_token = x[:, 0, :]
                patch_token = self.pooled(x)  # [batch, dim]
                z = self.projection_head(cls_token)
            else:
                cls_token = x[:, 0, :]  # CLS token [batch, dim]
                patch_token = x[:, 1:, :]
                z = self.projection_head(cls_token)
            return z, patch_token
            
        else:
            x = x.flatten(start_dim=1)  # For CNN like ResNet [batch, features]
            z = self.projection_head(x)
            return z, None
    
    def extract_features(self, x):
        x = self.backbone(x).flatten(start_dim=1)
        return x
        

import copy
from functools import partial

import torch
import torchvision
from timm.models.vision_transformer import vit_small_patch16_224
from torch import Tensor
from torch.nn import Module
from torch.optim import AdamW

from lightly.loss import DINOLoss, IBOTPatchLoss, KoLeoLoss
from lightly.models.modules import DINOv2ProjectionHead
from lightly.models.modules import MaskedVisionTransformerTIMM as origin_MaskedVisionTransformerTIMM
from lightly.models.utils import (
    random_block_mask,
    update_drop_path_rate,
    update_momentum,
)
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule, linear_warmup_schedule


def freeze_eval_module(module: Module) -> None:
    """Freeze the parameters of a module."""
    for param in module.parameters():
        param.requires_grad = False
    module.eval()


class DINOv2Head(Module):
    def __init__(
        self, dino_head: DINOv2ProjectionHead, ibot_head: DINOv2ProjectionHead
    ) -> None:
        super().__init__()
        self.dino_head = dino_head
        self.ibot_head = ibot_head


class DINOv2(Module):
    def __init__(
        self,
        ibot_separate_head: bool = False,
    ) -> None:
        super().__init__()

        # Backbones
        vit_teacher = vit_small_patch16_224(
            pos_embed="learn",
            dynamic_img_size=True,
            init_values=1e-5,
        )
        self.teacher_backbone = origin_MaskedVisionTransformerTIMM(
            vit=vit_teacher,
            antialias=False,
            pos_embed_initialization="skip",
        )
        self.student_backbone = copy.deepcopy(self.teacher_backbone)
        update_drop_path_rate(
            self.student_backbone.vit,
            drop_path_rate=0.1,  # we recommend using smaller rates like 0.1 for vit-s-14
            mode="uniform",
        )

        freeze_eval_module(self.teacher_backbone)

        # Heads
        dino_head = partial(
            DINOv2ProjectionHead,
            input_dim=384,
        )

        teacher_dino_head = dino_head()
        student_dino_head = dino_head()

        ibot_head = partial(
            DINOv2ProjectionHead,
            input_dim=384,
        )

        if ibot_separate_head:
            teacher_ibot_head = ibot_head()
            student_ibot_head = ibot_head()
        else:
            teacher_ibot_head = teacher_dino_head
            student_ibot_head = student_dino_head

        self.teacher_head = DINOv2Head(
            dino_head=teacher_dino_head,
            ibot_head=teacher_ibot_head,
        )
        self.student_head = DINOv2Head(
            dino_head=student_dino_head,
            ibot_head=student_ibot_head,
        )

        freeze_eval_module(self.teacher_head)

    def forward(self, x: Tensor) -> Tensor:
        return self.teacher_backbone(x)

    def forward_teacher(self, x: Tensor) -> tuple[Tensor, Tensor]:
        features = self.teacher_backbone.encode(x)
        cls_tokens = features[:, 0]
        return cls_tokens, features

    def forward_student(
        self, x: Tensor, mask: Tensor | None
    ) -> tuple[Tensor, Tensor | None]:
        features = self.student_backbone.encode(x, mask=mask)
        cls_tokens = features[:, 0]
        masked_features = None if mask is None else features[mask]
        return cls_tokens, masked_features