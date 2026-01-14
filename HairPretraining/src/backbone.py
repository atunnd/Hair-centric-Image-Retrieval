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
from lightly.models.modules import MAEDecoderTIMM, DenseCLProjectionHead
from lightly.transforms import MAETransform,  DenseCLTransform
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.models.modules.masked_vision_transformer_torchvision import (
    MaskedVisionTransformerTorchvision,
)

import torch.nn as nn

import torch
import torch.nn as nn
import torchvision.models as models
from .masked_vision_transformer_timm import MaskedVisionTransformerTIMM
from lightly.utils.scheduler import cosine_schedule
from lightly.models.modules.heads import MSNProjectionHead
from lightly.models.modules import MaskedVisionTransformerTorchvision

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

from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)

import torchvision
import torch
import torch.nn as nn
import copy

class MSN(nn.Module):
    def __init__(self, vit):
        super().__init__()

        self.mask_ratio = 0.15
        self.backbone = MaskedVisionTransformerTorchvision(vit=vit)
        self.projection_head = MSNProjectionHead(input_dim=768)

        self.anchor_backbone = copy.deepcopy(self.backbone)
        self.anchor_projection_head = copy.deepcopy(self.projection_head)

        utils.deactivate_requires_grad(self.backbone)
        utils.deactivate_requires_grad(self.projection_head)

        self.prototypes = nn.Linear(256, 1024, bias=False).weight

    def forward(self, images):
        out = self.backbone(images=images)
        return self.projection_head(out)

    def forward_masked(self, images):
        batch_size, _, _, width = images.shape
        seq_length = (width // self.anchor_backbone.vit.patch_size) ** 2
        idx_keep, _ = utils.random_token_mask(
            size=(batch_size, seq_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        out = self.anchor_backbone(images=images, idx_keep=idx_keep)
        return self.anchor_projection_head(out)
    
    @torch.no_grad()
    def extract_features(self, x):
        x = self.backbone(x)
        return x

class DenseCL(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.projection_head_global = DenseCLProjectionHead(2048, 2048, 512)
        self.projection_head_local = DenseCLProjectionHead(2048, 2048, 512)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_global_momentum = copy.deepcopy(
            self.projection_head_global
        )
        self.projection_head_local_momentum = copy.deepcopy(self.projection_head_local)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        utils.deactivate_requires_grad(self.backbone_momentum)
        utils.deactivate_requires_grad(self.projection_head_global_momentum)
        utils.deactivate_requires_grad(self.projection_head_local_momentum)

    def forward(self, x):
        query_features = self.backbone(x)
        query_global = self.pool(query_features).flatten(start_dim=1)
        query_global = self.projection_head_global(query_global)
        query_features = query_features.flatten(start_dim=2).permute(0, 2, 1)
        query_local = self.projection_head_local(query_features)
        # Shapes: (B, H*W, C), (B, D), (B, H*W, D)
        return query_features, query_global, query_local

    @torch.no_grad()
    def forward_momentum(self, x):
        key_features = self.backbone(x)
        key_global = self.pool(key_features).flatten(start_dim=1)
        key_global = self.projection_head_global(key_global)
        key_features = key_features.flatten(start_dim=2).permute(0, 2, 1)
        key_local = self.projection_head_local(key_features)
        return key_features, key_global, key_local

    def extract_features(self, x):
        x = self.backbone(x)
        return self.pool(x).flatten(start_dim=1)
    
    
from lightly.models.modules import BYOLPredictionHead, BYOLProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum

class BYOL(nn.Module):
    def __init__(self, backbone):
        super().__init__()

        self.backbone = backbone
        self.projection_head = BYOLProjectionHead(512, 1024, 256)
        self.prediction_head = BYOLPredictionHead(256, 1024, 256)

        self.backbone_momentum = copy.deepcopy(self.backbone)
        self.projection_head_momentum = copy.deepcopy(self.projection_head)

        deactivate_requires_grad(self.backbone_momentum)
        deactivate_requires_grad(self.projection_head_momentum)

    def forward(self, x):
        y = self.backbone(x).flatten(start_dim=1)
        z = self.projection_head(y)
        p = self.prediction_head(z)
        return p

    def forward_momentum(self, x):
        y = self.backbone_momentum(x).flatten(start_dim=1)
        z = self.projection_head_momentum(y)
        z = z.detach()
        return z



class SimCLR(nn.Module):
    def __init__(self, model=None):
        super().__init__()
        self.model = model
        print("Using backbone: ", self.model)
        
        # Handle backbone: For ViT, wrap it
        if "vit" in str(model):
            vit = models.vit_b_16()  # Load full ViT if backbone is not already wrapped
            self.backbone = ViTBackbone(vit)
            proj_input_dim = 768  # For vit_b_16
            output_dim = 512
        else:
            if model == "resnet18":
                backbone = torchvision.models.resnet18()
                self.backbone = nn.Sequential(*list(backbone.children())[:-1])
                proj_input_dim = 512
                output_dim=128
            elif model == "resnet50":
                backbone = torchvision.models.resnet50()
                self.backbone = nn.Sequential(*list(backbone.children())[:-1])
                proj_input_dim = 2048
                output_dim=1024
            else:
                raise ValueError("Unsupported model")
        
        self.projection_head = SimCLRProjectionHead(proj_input_dim, proj_input_dim, output_dim)
        
    def forward(self, x):
        x = self.backbone(x)  # ResNet: [batch, features, 1, 1]; ViT: [batch, seq_len, dim]
        
        if "vit" in str(self.model):
            cls_token = x[:, 0, :]  # CLS token [batch, dim]
            patch_token = x[:, 1:, :]
            z = self.projection_head(cls_token)
            return z, patch_token
        else:
            x = x.flatten(start_dim=1)  # For CNN like ResNet [batch, features]
            z = self.projection_head(x)
        return z, None
    
    def extract_features(self, x):
        features = self.backbone(x)  # ResNet: [batch, features, 1, 1]; ViT: [batch, seq_len, dim]
        
        if "vit" in str(self.model):
            embedding = features[:, 0, :]
        else:
            embedding = features.flatten(start_dim=1)  # [batch, features] for ResNet-like
        
        return embedding  # Always [batch, dim] raw from backbone
        

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
            decoder_depth=8,
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
    


from lightly.models.utils import (
    batch_shuffle,
    batch_unshuffle,
    deactivate_requires_grad,
    update_momentum,
)
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

class SimCLR(nn.Module):
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

    def forward(self, x):
        x = self.backbone(x)                # [batch, feat, 1, 1]
        x = x.flatten(start_dim=1)          # [batch, feat]
        z = self.projection_head(x)
        return z

    def extract_features(self, x):
        return self.backbone(x).flatten(start_dim=1)
        
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import vit_base_patch32_224
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM


import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from timm.models.vision_transformer import vit_base_patch16_224
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM


import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from timm.models.vision_transformer import vit_base_patch16_224
from lightly.models import utils

import torch
import torch.nn as nn
import torch.nn.functional as F

class CrossAlignBlock(nn.Module):
    """
    Cross-Attention Alignment Block:
    Student embeddings attend to Teacher embeddings.
    """
    def __init__(self, dim, num_heads=8, mlp_ratio=4.0):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, E_s, E_t):
        # Cross-attention: Q = student, K/V = teacher
        attn_out, _ = self.cross_attn(
            query=E_s, key=E_t, value=E_t
        )
        E_s = self.norm1(E_s + attn_out)  # residual + norm
        E_s = self.norm2(E_s + self.mlp(E_s))  # feed-forward
        return E_s

class PosMapping(nn.Module):
    """Learnable mapping from student PE → teacher PE space."""
    def __init__(self, dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, pe):
        return self.fc(pe)


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

        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.patch_size = vit.patch_embed.patch_size[0]
        embed_dim = vit.embed_dim

        #print("=>> pos_embed: ", vit.pos_embed.shape) # [1, 197, 768]

        if mode == "embedding":
            decoder_dim = embed_dim
        elif mode == "reconstruction":
            decoder_dim = decoder_dim

        # ========== Decoder ==========
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=2,
            decoder_num_heads=8,
            mlp_ratio=4.0,
        )

        # =========== Cross Alignment ===========
        self.cross_align = CrossAlignBlock(embed_dim, num_heads=8)
        self.pos_map = PosMapping(embed_dim)

        # ========== Dual Projection Heads ==========
        # self.proj_global = nn.Sequential(
        #     nn.Linear(embed_dim, decoder_dim),
        #     nn.LayerNorm(decoder_dim),
        #     nn.GELU(inplace=True),
        #     nn.Linear(decoder_dim, decoder_dim),
        # )
        self.proj_global = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 256),
        )

        # self.proj_local = nn.Sequential(
        #     nn.Linear(embed_dim, decoder_dim),
        #     nn.LayerNorm(decoder_dim),
        #     nn.GELU(inplace=True),
        #     nn.Linear(decoder_dim, decoder_dim),
        # )
        self.proj_local = nn.Sequential(
            nn.Linear(embed_dim, 1024), # decoder_dim = 768
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Linear(1024, 512),
        )

        # ========== Optional Attention Pooling ==========
        if pooling == "attention":
            self.att_pool = nn.MultiheadAttention(
                embed_dim=embed_dim, num_heads=8, batch_first=True
            )

        # ========== EMA Teacher ==========
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_proj_global = copy.deepcopy(self.proj_global)
        self.teacher_proj_local = copy.deepcopy(self.proj_local)

        for p in (
            list(self.teacher_backbone.parameters())
            + list(self.teacher_proj_global.parameters())
            + list(self.teacher_proj_local.parameters())
        ):
            p.requires_grad = False


    # ---------------- Encoder ----------------
    def forward_encoder(self, images, idx_keep=None, idx_mask=None):
        if self.mode == "reconstruction":
          x_encoded = self.backbone.encode(images=images, idx_keep=idx_keep)
        elif self.mode == "embedding":
          x_encoded = self.backbone.encode(images=images, idx_keep=None, idx_mask=idx_mask)

        cls_token = x_encoded[:, 0]

        if self.mode == "embedding": # output patches included visable tokens + masked tokens
          visable_tokens = get_at_index(x_encoded, idx_keep)
          masked_tokens = get_at_index(x_encoded, idx_mask) # masked tokens
        elif self.mode == "reconstruction":
          visable_tokens = x_encoded[:, 1:]

        patch_tokens = x_encoded[:, 1:]
          
        if self.pooling == "mean":
            pooled = visable_tokens.mean(dim=1) # global contrastive 

        #patch_tokens = patch_tokens.mean(dim=1)

        return x_encoded, cls_token, pooled, patch_tokens
    
    def forward_encoder_teacher(self, images, idx_keep=None, idx_mask=None):
        if self.mode == "reconstruction":
          x_encoded = self.teacher_backbone.encode(images=images, idx_keep=idx_keep)
        elif self.mode == "embedding":
          x_encoded = self.teacher_backbone.encode(images=images, idx_keep=None)

        cls_token = x_encoded[:, 0]
        patch_tokens = x_encoded[:, 1:]

        if self.pooling == "mean":
            pooled = patch_tokens.mean(dim=1) # global contrastive 

        #patch_tokens = patch_tokens.mean(dim=1)

        return x_encoded, cls_token, pooled, patch_tokens

    # ---------------- Decoder ----------------
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

    # ---------------- Full Forward ----------------
    def forward(self, img_student, img_teacher=None):
        """
        Unified forward supporting:
          - mode == "reconstruction": student reconstructs pixel patches for masked patches.
          - mode == "embedding": student predicts masked patch embeddings; teacher is full-view EMA.
        """
        if img_teacher is None:
            img_teacher = img_student

        B = img_student.shape[0]
        idx_keep_patches, idx_mask_patches = utils.random_token_mask(
            size=(B, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=img_student.device,
        )

        # ---------- Student Forward (masked) ----------
        x_enc_s, cls_s, pooled_s, patch_s = self.forward_encoder(img_student, idx_keep=idx_keep_patches, idx_mask=idx_mask_patches)
        student_global = self.proj_global(pooled_s)
        
        if self.mode == "reconstruction":
          x_pred_s = self.forward_decoder(x_enc_s, idx_keep=idx_keep_patches, idx_mask=idx_mask_patches)
          N_total = patch_s.shape[1] + x_pred_s.shape[1]
          x_pred_s = self.merge_visible_and_masked(
              patch_vis=patch_s,
              patch_mask=x_pred_s,
              idx_keep=idx_keep_patches[:, 1:]-1,
              idx_mask=idx_mask_patches[:, 1:]-1,
              N_total=N_total
          )
        elif self.mode == "embedding":
          x_pred_s = patch_s


        # ---------- Teacher Forward (full view, no mask) ----------
        with torch.no_grad():
            x_enc_t, cls_t, pooled_t, patch_t = self.forward_encoder_teacher(img_teacher, idx_keep=None, idx_mask=idx_mask_patches)
            teacher_global = self.teacher_proj_global(pooled_t)
            teacher_local = self.teacher_proj_local(patch_t)

        # ---------- Cross Alignemnt between Student and Teacher ---------
        # Add positional embeddings
        x_pred_s  = x_pred_s  + self.pos_map(self.backbone.vit.pos_embed[:, 1:, :])
        patch_t = patch_t + self.teacher_backbone.vit.pos_embed[:, 1:, :]

        # Cross-alignment
        x_pred_s_refined = self.cross_align(x_pred_s, patch_t)
        student_local = self.proj_local(x_pred_s_refined) 

        return {
                "global_s": student_global,
                "global_t": teacher_global,
                "local_s": student_local,
                "local_t": teacher_local,
                "idx_keep": idx_keep_patches,
                "idx_mask": idx_mask_patches,
            }

    def merge_visible_and_masked(self, patch_vis, patch_mask, idx_keep, idx_mask, N_total):
        """
        patch_vis: [B, N_vis, D]
        patch_mask: [B, N_mask, D]
        idx_keep: [B, N_vis] (long)
        idx_mask: [B, N_mask] (long)
        N_total: tổng số patch ban đầu (H_p * W_p)
        """
        B, D = patch_vis.shape[0], patch_vis.shape[-1]
        device = patch_vis.device
        
        # tensor chứa kết quả
        merged = torch.zeros(B, N_total, D, device=device, dtype=patch_vis.dtype)
        
        # ghép visible patch vào vị trí đúng
        merged.scatter_(1, idx_keep.unsqueeze(-1).expand(-1, -1, D), patch_vis)
        
        # ghép masked patch vào vị trí đúng
        merged.scatter_(1, idx_mask.unsqueeze(-1).expand(-1, -1, D), patch_mask)
        
        return merged  # [B, N_total, D]

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

    def forward_teacher(self, x: Tensor):
        features = self.teacher_backbone.encode(x)
        cls_tokens = features[:, 0]
        return cls_tokens, features

    def forward_student(
        self, x: Tensor, mask: Tensor 
    ):
        features = self.student_backbone.encode(x, mask=mask)
        cls_tokens = features[:, 0]
        masked_features = None if mask is None else features[mask]
        return cls_tokens, masked_features

    def extract_features(self, x):
        features = self.student_backbone.encode(x)
        # lấy CLS token (B, D)
        cls_tokens = features[:, 0]
        return cls_tokens

class ViTFeatureExtractor:
    def __init__(self, model):
        self.model = model
        self.model.eval()

    def extract_features(self, x):
        with torch.no_grad():
            features = self.model.forward_features(x)
            return features[:, 0]  # CLS token


# ------------------------------------------------------------------------
# SiameseIM
# Copyright (c) SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from MAE (https://github.com/facebookresearch/mae)
# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved.
# ------------------------------------------------------------------------
# References:
# timm: https://github.com/rwightman/pytorch-image-models/tree/master/timm
# DeiT: https://github.com/facebookresearch/deit
# ------------------------------------------------------------------------


import random
from functools import partial
from turtle import update
import math


import torch
import torch.nn as nn

from .models_vit import Block, CrossBlock, PatchEmbed 
import numpy as np

class LayerNorm(nn.LayerNorm):

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, input):
        return super(LayerNorm, self).forward(input.float())

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


class PermuteBN(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bn = nn.BatchNorm1d(dim)

    @torch.cuda.amp.autocast(enabled=False)
    def forward(self, x):
        x = x.permute(0, 2, 1) # N, L, C -> N, C, L
        x = x.float()
        x = self.bn(x)
        x = x.permute(0, 2, 1) # N, C, L -> N, L, C

        return x


class SiameseIMViT(nn.Module):
    """  SiameseIM with VisionTransformer backbone
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=1024, depth=24, num_heads=16,
                 decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
                 mlp_ratio=4., norm_layer=LayerNorm, norm_pix_loss=False, args=None):
        super().__init__()
        self.norm_pix_loss = norm_pix_loss
        self.args = args
        decoder_embed_dim = args.decoder_embed_dim

        # --------------------------------------------------------------------------
        # encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.num_patches = num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        if args.use_abs_pos_emb:
            if hasattr(self, 'cls_token'):
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, embed_dim), requires_grad=False)  # fixed sin-cos embedding
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=False)  # fixed sin-cos embedding

        dpr = [x.item() for x in torch.linspace(0, args.drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer,
                    drop_path=dpr[i], init_values=args.init_values)
            for i in range(depth)])
        # --------------------------------------------------------------------------

        # --------------------------------------------------------------------------
        # decoder specifics
        if args.loss_type in ['mae']:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            if hasattr(self, 'cls_token'):
                self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
            else:
                self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)

            self.decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer)
                for i in range(decoder_depth)])

            self.decoder_norm = norm_layer(decoder_embed_dim)
            self.decoder_pred = nn.Linear(decoder_embed_dim, patch_size**2 * in_chans, bias=True) # decoder to patch
        elif args.loss_type in ['sim',]:
            self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

            if hasattr(self, 'cls_token'):
                self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches+1, decoder_embed_dim), requires_grad=False)  # fixed sin-cos embedding
            else:
                self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim), requires_grad=False)

            if args.projector_depth > 0:
                self.projector_decoder_blocks = nn.ModuleList([
                    Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                          norm_layer=norm_layer if args.use_proj_ln else PermuteBN)
                    for i in range(args.projector_depth)])

            self.predictor_decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                      norm_layer=norm_layer if args.use_pred_ln else PermuteBN)
                for i in range(args.predictor_depth)])

            self.decoder_pred = nn.Linear(decoder_embed_dim, decoder_embed_dim, bias=True) # decoder to patch
            if args.online_ln:
                self.student_norm = LayerNorm(decoder_embed_dim)
                for p in self.student_norm.parameters():
                    p.requires_grad = False
            else:
                self.student_norm = nn.Identity()
        # --------------------------------------------------------------------------
    
        # ---------------------------------------------------------------------------
        # decoder pos embed change dim
        if self.args.loss_type in ['sim',]:
            self.decoder_pos_mlp = nn.Linear(decoder_embed_dim*2, decoder_embed_dim)
        # ---------------------------------------------------------------------------

        self.initialize_weights()

        # build momentum branch
        if self.args.loss_type in ['sim',]:
            self.build_momentum_target(img_size, patch_size, in_chans, embed_dim, num_heads,
                                        mlp_ratio, norm_layer, depth, decoder_embed_dim, decoder_num_heads)

        # stop grad for patch embedding
        if (not args.train_patch_embed):
            self.patch_embed.proj.weight.requires_grad = False
            self.patch_embed.proj.bias.requires_grad = False

    def build_momentum_target(self, img_size, patch_size, in_chans, embed_dim, num_heads,
                                mlp_ratio, norm_layer, depth, decoder_embed_dim, decoder_num_heads):
        # --------------------------------------------------------------------------
        # momentum encoder specifics
        self.mm_patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)

        if hasattr(self, 'cls_token'):
            self.mm_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.mm_blocks = nn.ModuleList([
            Block(embed_dim, num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, init_values=self.args.init_values)
            for i in range(depth)])
        
        # load weight
        self.mm_patch_embed.load_state_dict(self.patch_embed.state_dict())
        for p in self.mm_patch_embed.parameters():
            p.requires_grad = False

        self.mm_cls_token.data.copy_(self.cls_token.data)
        self.mm_cls_token.requires_grad = False

        self.mm_blocks.load_state_dict(self.blocks.state_dict())
        for p in self.mm_blocks.parameters():
            p.requires_grad = False
        # --------------------------------------------------------------------------
 
        # --------------------------------------------------------------------------
        # momentum decoder specifics
        self.mm_decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mm_mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if self.args.projector_depth > 0:
            self.mm_projector_decoder_blocks = nn.ModuleList([
                Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer if self.args.use_proj_ln else PermuteBN)
                for i in range(self.args.projector_depth)])

        # load weight
        self.mm_decoder_embed.load_state_dict(self.decoder_embed.state_dict())
        for p in self.mm_decoder_embed.parameters():
            p.requires_grad = False
        
        self.mm_mask_token.data.copy_(self.mask_token.data)
        self.mm_mask_token.requires_grad = False
        
        if self.args.projector_depth > 0:
            self.mm_projector_decoder_blocks.load_state_dict(self.projector_decoder_blocks.state_dict())
            for p in self.mm_projector_decoder_blocks.parameters():
                p.requires_grad = False
        # ---------------------------------------------------------------------------

        if self.args.loss_type in ['sim',]:
            self.teacher_norm = LayerNorm(decoder_embed_dim, elementwise_affine=False)
            for p in self.teacher_norm.parameters():
                p.requires_grad = False

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        if self.args.use_abs_pos_emb:
            pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.num_patches**.5), cls_token=hasattr(self, 'cls_token'))
            self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        if hasattr(self, 'decoder_pos_embed'):
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1], int(self.num_patches**.5), cls_token=hasattr(self, 'cls_token'))
            self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        if hasattr(self, 'cls_token'):
            torch.nn.init.normal_(self.cls_token, std=.02)
        if hasattr(self, 'mask_token'):
            torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 3, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2 * 3))
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1]**.5)
        assert h * w == x.shape[1]
        
        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 3, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        # ids_keep = ids_shuffle[:, :len_keep]
        # x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return mask, ids_restore

    @torch.cuda.amp.autocast(enabled=False)
    def mm_update(self, mm):
        for param_q, param_k in zip(self.patch_embed.parameters(), self.mm_patch_embed.parameters()):
            param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        for param_q, param_k in zip(self.blocks.parameters(), self.mm_blocks.parameters()):
            param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_cls_token'):
            self.mm_cls_token.data = self.mm_cls_token.data * mm + self.cls_token.data * (1. - mm)
        if hasattr(self, 'mm_norm'):
            for param_q, param_k in zip(self.norm.parameters(), self.mm_norm.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_projector'):
            for param_q, param_k in zip(self.projector.parameters(), self.mm_projector.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_decoder_embed'):
            for param_q, param_k in zip(self.decoder_embed.parameters(), self.mm_decoder_embed.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_mask_token'):
            self.mm_mask_token.data = self.mm_mask_token.data * mm + self.mask_token.data * (1. - mm)
        if hasattr(self, 'mm_decoder_blocks'):
            for param_q, param_k in zip(self.decoder_blocks.parameters(), self.mm_decoder_blocks.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_projector_decoder_blocks'):
            for param_q, param_k in zip(self.projector_decoder_blocks.parameters(), self.mm_projector_decoder_blocks.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_decoder_norm'):
            for param_q, param_k in zip(self.decoder_norm.parameters(), self.mm_decoder_norm.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)
        if hasattr(self, 'mm_decoder_pred'):
            for param_q, param_k in zip(self.decoder_pred.parameters(), self.mm_decoder_pred.parameters()):
                param_k.data = param_k.data * mm + param_q.data * (1. - mm)

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed

        # masking: length -> length * mask_ratio
        mask, ids_restore = self.random_masking(x, mask_ratio)
        x = x[~mask.bool()].view(x.shape[0], -1, x.shape[-1])

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x, mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        x = x_

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*3]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward_mae(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask
    
    def forward(self, *args, **kwargs):
        if self.args.loss_type in ['sim',]:
            return self.forward_sim(*args, **kwargs)
        else:
            return self.forward_mae(*args, **kwargs)

    def forward_sim(self, x1, x2, rel_pos_21, mm, update_mm, mask=None):
        # forward online encoder
        if self.args.with_blockwise_mask:
            assert mask is not None, 'mask should not be None when mask_type is block'
            mask = mask.view(mask.shape[0], -1)
        else:
            assert False
            mask, ids_restore1 = self.random_masking(online_x1, self.args.mask_ratio)
        
        online_x1 = self.patch_embed(x1)
        online_x1 = online_x1 + self.pos_embed[:, 1:, :]
        online_x1 = online_x1[~mask.bool()].view(online_x1.shape[0], -1, online_x1.shape[-1])
        # add cls token
        cls_tokens = self.cls_token.expand(online_x1.shape[0], -1, -1) + self.pos_embed[:, 0, :].unsqueeze(1)
        online_x1 = torch.cat((cls_tokens, online_x1), dim=1)

        # forward online encoder
        for blk in self.blocks:
            online_x1 = blk(online_x1)

        # forward online projector
        online_x1 = self.decoder_embed(online_x1)
        if self.args.projector_depth > 0:
            for blk in self.projector_decoder_blocks:
                online_x1 = blk(online_x1)

        # calculate decoder pos embed
        cls_pos_embed = self.decoder_pos_embed[:, 0, :].unsqueeze(1)
        x1_vis_embed = self.decoder_pos_embed[:, 1:, :].repeat(online_x1.shape[0], 1, 1)[~mask.bool()].view(online_x1.shape[0], -1, self.decoder_pos_embed.shape[-1])
        x2_embed = get_2d_sincos_pos_embed_relative(*rel_pos_21, self.decoder_pos_embed.shape[-1],
                                                    int(self.num_patches ** .5))
        x2_embed = self.decoder_pos_mlp(x2_embed)

        # append mask tokens to sequence
        cls_token = online_x1[:, 0, :].unsqueeze(1)
        x1_vis_tokens = online_x1[:, 1:, :]
        mask_tokens = self.mask_token.repeat(x2.shape[0], x2_embed.shape[1], 1)
        x = torch.cat([cls_token+cls_pos_embed, x1_vis_tokens+x1_vis_embed, mask_tokens+x2_embed], dim=1)

        # forward online decoder
        for blk in self.predictor_decoder_blocks:
            x = blk(x)

        # predictor projection
        x = self.decoder_pred(x)
        pred = x[:, -x2_embed.shape[1]:]

        # forward target encoder
        with torch.no_grad():
            if update_mm:
                self.mm_update(mm)

            target_x2 = self.mm_patch_embed(x2)
            mm_cls_tokens = self.mm_cls_token.expand(target_x2.shape[0], -1, -1)
            target_x2 = torch.cat((mm_cls_tokens, target_x2), dim=1)
            target_x2 = target_x2 + self.pos_embed

            # forward target encoder
            for blk in self.mm_blocks:
                target_x2 = blk(target_x2)

            # forward target projector
            target_x2 = self.mm_decoder_embed(target_x2)
            if self.args.projector_depth > 0:
                for blk in self.mm_projector_decoder_blocks:
                    target_x2 = blk(target_x2)

            target = target_x2[:, 1:, :]

        # compute loss
        outputs = {}
        with torch.cuda.amp.autocast(enabled=False):
            loss = self.compute_unigrad_loss(pred.float(), target.float())
        outputs['loss_sim'] = loss.item()

        return loss, outputs

    def compute_unigrad_loss(self, pred, target):
        pred = self.student_norm(pred)
        with torch.no_grad():
            target = self.teacher_norm(target)
        
        dense_pred = pred.reshape(-1, pred.shape[-1])
        dense_target = target.reshape(-1, target.shape[-1])

        # compute pos term
        pos_term = ((dense_pred - dense_target)**2).sum(-1).mean()

        # compute neg term
        correlation = (dense_target.T @ dense_target) / dense_target.shape[0]
        torch.distributed.all_reduce(correlation)
        correlation = correlation / torch.distributed.get_world_size()
        
        neg_term = torch.diagonal(dense_pred @ correlation @ dense_pred.T).mean()

        loss = (pos_term + self.args.neg_weight * neg_term) / pred.shape[-1]

        return loss
    
    # def extract_features(self, x, return_cls=True):
    #     """
    #     Extract embeddings from SiameseIMViT (loss_type='sim') for inference/retrieval.
        
    #     Args:
    #         x (Tensor): Input batch, shape (B, C, H, W)
    #         return_cls (bool): Nếu True -> trả về embedding cls token,
    #                         Nếu False -> trả về toàn bộ patch embeddings.
        
    #     Returns:
    #         Tensor: Embeddings (B, D) nếu return_cls=True, 
    #                 hoặc (B, N, D) nếu return_cls=False.
    #     """
    #     # patchify + pos embed
    #     x = self.patch_embed(x)   # (B, N, D)
    #     # thêm cls token
    #     cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
    #     x = torch.cat((cls_tokens, x), dim=1)
    #     x = x + self.pos_embed

    #     # forward qua backbone transformer
    #     for blk in self.blocks:
    #         x = blk(x)
    #     if hasattr(self, "norm"):
    #         x = self.norm(x)
    

        # # projector để ra embedding cuối
        # x = self.decoder_embed(x)
        # if self.args.projector_depth > 0:
        #     for blk in self.projector_decoder_blocks:
        #         x = blk(x)

        # if return_cls:
        #     return x[:, 0]   # lấy cls token embedding (B, D)
        # else:
        #     return x 
    def extract_features(self, x1, mask=None):
        # patch embedding
        online_x1 = self.patch_embed(x1)
        online_x1 = online_x1 + self.pos_embed[:, 1:, :]
        
        if mask is not None:  # nếu muốn dùng mask
            online_x1 = online_x1[~mask.bool()].view(online_x1.shape[0], -1, online_x1.shape[-1])
        else:  # không mask thì lấy full
            pass

        # thêm cls token
        cls_tokens = self.cls_token.expand(online_x1.shape[0], -1, -1) + self.pos_embed[:, 0, :].unsqueeze(1)
        online_x1 = torch.cat((cls_tokens, online_x1), dim=1)

        # forward qua các transformer block
        for blk in self.blocks:
            online_x1 = blk(online_x1)

        # chỉ lấy cls token (B, D)
        cls_feature = online_x1[:, 0, :]
        return cls_feature



def sim_vit_base_patch16_dec512d8b(**kwargs):
    model = SiameseIMViT(
        patch_size=16, embed_dim=768, depth=12, num_heads=12,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(LayerNorm, eps=1e-6), **kwargs)
    return model


def sim_vit_large_patch16_dec512d8b(**kwargs):
    model = SiameseIMViT(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(LayerNorm, eps=1e-6), **kwargs)
    return model


def sim_vit_huge_patch14_dec512d8b(**kwargs):
    model = SiameseIMViT(
        patch_size=14, embed_dim=1280, depth=32, num_heads=16,
        decoder_embed_dim=512, decoder_depth=8, decoder_num_heads=16,
        mlp_ratio=4, norm_layer=partial(LayerNorm, eps=1e-6), **kwargs)
    return model


# set recommended archs
sim_vit_base_patch16 = sim_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
sim_vit_large_patch16 = sim_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
sim_vit_huge_patch14 = sim_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

