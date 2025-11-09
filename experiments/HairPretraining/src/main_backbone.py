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

class ProjectionHead(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=2048, output_dim=128, use_bn=True):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim) if use_bn else nn.Identity()
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.bn2 = nn.BatchNorm1d(output_dim) if use_bn else nn.Identity()

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        # L2 normalize for contrastive loss
        x = F.normalize(x, dim=-1)
        return x

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
        self.embed_dim = embed_dim

        #print("=>> pos_embed: ", vit.pos_embed.shape) # [1, 197, 768]

        if mode == "embedding":
            decoder_dim = embed_dim
        elif mode == "reconstruction":
            decoder_dim = decoder_dim

        # ========== Decoder ==========
        self.pixel_decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=8,
            decoder_num_heads=8,
            mlp_ratio=4.0,
        )
        
        self.embedding_decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=2,
            decoder_num_heads=8,
            mlp_ratio=4.0,
        )

        self.proj_head = ProjectionHead(input_dim=embed_dim)

        # ========== EMA Teacher ==========
        self.teacher_backbone = copy.deepcopy(self.backbone)
        self.teacher_proj_head = copy.deepcopy(self.proj_head)

        for p in (
            list(self.teacher_backbone.parameters())
            + list(self.teacher_proj_head.parameters())
        ):
            p.requires_grad = False


    # ---------------- Encoder ----------------
    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)
    
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
        
        # ----- forward anchor ------
        x_encoded = self.forward_encoder(images=img_anchor, idx_keep=idx_keep)
        x_pred_pixel = self.forward_pixel_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )
        x_pred_pixel = self.forward_embedding_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )
        anchor_embedding = self.proj_head(x_pred_pixel[:, 1:, :].mean(dim=1))

        # get image patches for masked tokens
        patches = utils.patchify(img_anchor, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)
        
        # ------ forward pos1 ------
        pos_encoded_1 = self.forward_encoder(images=img_pos1, idx_keep=None)
        pos1_embedding = self.proj_head(pos_encoded_1[:, 1:, :].mean(dim=1))
        
        # ------ forward pos2 ------
        with torch.no_grad():
            pos_encoded_2 = self.forward_encoder_teacher(images=img_pos2, idx_keep=None)
            pos2_embedding = self.teacher_proj_head(pos_encoded_2[:, 1:, :].mean(dim=1))
        
        return {
            "anchor": anchor_embedding,
            "pos1": pos1_embedding,
            "pos2": pos2_embedding,
            'masked_prediction': x_pred,
            'masked_GT': target
        }
        
    def extract_features(self, images):
        x_encoded = self.forward_encoder(images)
        return x_encoded[:, :1].mean(dim=1)


import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer
from torch import Tensor
from torch.nn import LayerNorm, Linear, Module, Parameter

from lightly.models import utils


model_args = dict(patch_size=16, embed_dim=768, depth=12, num_heads=12)

from abc import ABC, abstractmethod
from typing import List, Optional, Tuple

from torch import Tensor
from torch.nn import Module, Parameter


class MaskedVisionTransformer(ABC, Module):
    """
    Abstract base class for Masked Vision Transformer models.

    Defines the interface for a Masked Vision Transformer. This class includes abstract
    methods that must be implemented by concrete subclasses to define the forward pass,
    tokenization of images, and various operations needed for the transformer.
    """

    # This is not defined as a property for backwards compatibility.
    # New models should define this as a property.
    mask_token: Parameter

    @property
    @abstractmethod
    def sequence_length(self) -> int:
        ...

    @abstractmethod
    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Returns encoded class tokens from a batch of images.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_size, image_size).
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, the indexed tokens are masked with self.mask_token.
                Cannot be used in combination with mask argument.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, only the indexed tokens will be forwarded.
                Is applied after any masking operation.
            mask:
                Boolean tensor with shape (batch_size, sequence_length) indicating
                which tokens should be masked. Tokens where the mask is True will be
                replaced with the mask token.
                Cannot be used in combination with idx_mask argument.

        Returns:
            Tensor with shape (batch_size, embed_dim) containing the encoded class token
            for every image.

        """
        ...

    @abstractmethod
    def forward_intermediates(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        norm: bool = False,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        """Encode input images and return features from the intermediate layers.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_height, image_width).
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                Indices must be in the range [0, sequence_length).
                If specified, the indexed tokens are masked with self.mask_token.
                Cannot be used in combination with mask argument.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, only the indexed tokens will be forwarded.
                Is applied after any masking operation.
            norm:
                Apply norm layer to all intermediates.
            mask:
                Boolean tensor with shape (batch_size, sequence_length) indicating
                which tokens should be masked. Tokens where the mask is True will be
                replaced with the mask token.
                Cannot be used in combination with idx_mask argument.

        Returns:
            Tuple of batch of encoded output tokens and a list of intermediate features.
            The encoded output tokens have shape (batch_size, embed_dim) and each
            intermediate feature has shape (batch_size, sequence_length, embed_dim).
            If idx_keep is set, only num_tokens_to_keep tokens per sequence are
            returned.
        """
        ...

    @abstractmethod
    def encode(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode input images.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_height, image_width).
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                Indices must be in the range [0, sequence_length).
                If specified, the indexed tokens are masked with self.mask_token.
                Cannot be used in combination with mask argument.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, only the indexed tokens will be encoded.
                Is applied after any masking operation.
            mask:
                Boolean tensor with shape (batch_size, sequence_length) indicating
                which tokens should be masked. Tokens where the mask is True will be
                replaced with the mask token.
                Cannot be used in combination with idx_mask argument.

        Returns:
            Tensor with shape (batch_size, sequence_length, embed_dim) containing the
            encoded output tokens. If idx_keep is set, only num_tokens_to_keep tokens
            per sequence are returned.
        """
        ...

    def preprocess(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Convert images to tokens, add positional embeddings, and apply masking.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_height, image_width).
            idx_mask:
                Tensor with shape (batch_size, num_tokens_to_mask) where each
                entry is an index of the token to mask in the respective batch.
                Indices must be in the range [0, sequence_length).
                If specified, the indexed tokens are masked with self.mask_token.
                Cannot be used in combination with mask argument.
            idx_keep:
                Tensor with shape (batch_size, num_tokens_to_keep) where each
                entry is an index of the token to keep in the respective batch.
                Indices must be in the range [0, sequence_length).
                If set, only the indexed tokens will be returned.
                Is applied after any masking operation.
            mask:
                Tensor with shape (batch_size, sequence_length) indicating which tokens
                should be masked. Tokens where the mask is True will be masked with
                self.mask_token.

        Returns:
            Tensor with shape (batch_size, sequence_length, embed_dim) containing the
            preprocessed tokens. If idx_keep is set, only num_tokens_to_keep tokens
            per sequence are returned. Any class or prefix tokens are prepended to the
            sequence.
        """
        if idx_mask is not None and mask is not None:
            raise ValueError("idx_mask and mask cannot both be set at the same time.")

        # convert images to tokens
        tokens = self.images_to_tokens(images)
        # add prefix tokens if needed
        tokens = self.prepend_prefix_tokens(tokens)

        if idx_mask is not None:
            tokens = utils.mask_at_index(
                tokens=tokens, index=idx_mask, mask_token=self.mask_token
            )
        elif mask is not None:
            tokens = utils.mask_bool(
                tokens=tokens, mask=mask, mask_token=self.mask_token
            )

        # add positional encoding
        tokens = self.add_pos_embed(tokens)

        if idx_keep is not None:
            tokens = utils.get_at_index(tokens, idx_keep)

        return tokens

    @abstractmethod
    def images_to_tokens(self, images: Tensor) -> Tensor:
        """Converts images into patch tokens.

        Args:
            images:
                Tensor with shape (batch_size, channels, image_height, image_width).

        Returns:
            Tensor with shape (batch_size, num_patches, embed_dim) containing the
            patch tokens (excluding prefix tokens).
        """
        ...

    # Keep for backwards compatibility.
    def add_prefix_tokens(self, x: Tensor) -> Tensor:
        return self.prepend_prefix_tokens(x)

    @abstractmethod
    def prepend_prefix_tokens(self, x: Tensor) -> Tensor:
        """Prepends prefix tokens to the input patch tokens.

        Args:
            x:
                Tensor with shape (batch_size, num_patches, embed_dim) containing patch
                tokens.

        Returns:
            Tensor with shape (batch_size, sequence_length, embed_dim) containing
            the prefix and patch tokens. The prefix tokens are prepended to the
            sequence.
        """
        ...

    @abstractmethod
    def add_pos_embed(self, x: Tensor) -> Tensor:
        """Adds positional embeddings to the input tokens.

        Args:
            x:
                Tensor with shape (batch_size, sequence_length, embed_dim) containing
                the input tokens. Must include prefix tokens.

        Returns:
            Tensor after adding positional embeddings, with the same shape as the input.
        """
        


class MaskedVisionTransformerTIMM(MaskedVisionTransformer):
    """Masked Vision Transformer class using TIMM.

    Attributes:
        vit:
            The VisionTransformer object of TIMM.
        mask_token:
            The mask token.
        weight_initialization:
            The weight initialization method. Valid options are ['', 'skip']. '' uses
            the default MAE weight initialization and 'skip' skips the weight
            initialization.
        antialias:
            Whether to use antialiasing when resampling the positional embeddings.
        pos_embed_initialization:
            The strategy to initialize the positional embeddings. Valid options are
            ['learn', 'sincos', 'skip'].

    """

    def __init__(
        self,
        vit: VisionTransformer,
        mask_token: Optional[Parameter] = None,
        weight_initialization: str = "",
        antialias: bool = True,
        pos_embed_initialization: str = "sincos",
    ) -> None:
        super().__init__()
        self.vit = vit
        self.mask_token = (
            mask_token
            if mask_token is not None
            else Parameter(torch.zeros(1, 1, self.vit.embed_dim))
        )

        if weight_initialization not in ("", "skip"):
            raise ValueError(
                f"Invalid weight initialization method: '{weight_initialization}'. "
                "Valid options are: ['', 'skip']."
            )
        if weight_initialization != "skip":
            self._initialize_weights()

        utils.initialize_positional_embedding(
            pos_embedding=self.vit.pos_embed,
            strategy=pos_embed_initialization,
            num_prefix_tokens=self.vit.num_prefix_tokens,
        )

        self.antialias = antialias

    @property
    def sequence_length(self) -> int:
        seq_len: int = self.vit.patch_embed.num_patches + self.vit.num_prefix_tokens
        return seq_len

    def forward(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        x = self.encode(images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask)
        if self.vit.attn_pool is not None:
            x = self.vit.attn_pool(x)
        elif self.vit.global_pool == "avg":
            x = x[:, self.vit.num_prefix_tokens :].mean(dim=1)
        elif self.vit.global_pool:
            x = x[:, 0]  # class token
        return x

    def forward_intermediates(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        norm: bool = False,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, List[Tensor]]:
        # preprocess images, convert to tokens and add positional embeddings
        tokens = self.preprocess(
            images=images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask
        )
        # normalization layer
        tokens = self.vit.norm_pre(tokens)

        intermediates: List[Tensor] = []
        for blk in self.vit.blocks:
            tokens = blk(tokens)
            intermediates.append(self.vit.norm(tokens) if norm else tokens)

        # normalize
        out: Tensor = self.vit.norm(tokens)

        return out, intermediates

    def encode(
        self,
        images: Tensor,
        idx_mask: Optional[Tensor] = None,
        idx_keep: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        # preprocess images, convert to tokens and add positional embeddings
        tokens: Tensor = self.preprocess(
            images=images, idx_mask=idx_mask, idx_keep=idx_keep, mask=mask
        )
        # normalization layer
        tokens = self.vit.norm_pre(tokens)
        # apply Transformer blocks
        tokens = self.vit.blocks(tokens)
        # normalize
        tokens = self.vit.norm(tokens)
        return tokens

    def images_to_tokens(self, images: Tensor) -> Tensor:
        tokens: Tensor = self.vit.patch_embed(images)
        if self.vit.dynamic_img_size:
            tokens = tokens.permute(0, 3, 1, 2)  # NHWC -> NCHW
            tokens = tokens.flatten(2).transpose(1, 2)  # NCHW -> NLC
        return tokens

    def prepend_prefix_tokens(self, x: Tensor) -> Tensor:
        prefix_tokens = []
        if self.vit.cls_token is not None:
            prefix_tokens.append(self.vit.cls_token.expand(x.shape[0], -1, -1))
        if self.vit.reg_token is not None:
            prefix_tokens.append(self.vit.reg_token.expand(x.shape[0], -1, -1))
        if prefix_tokens:
            x = torch.cat(prefix_tokens + [x], dim=1)
        return x

    def add_pos_embed(self, x: Tensor) -> Tensor:
        x_prefix = x[:, : self.vit.num_prefix_tokens, :]
        x = x[:, self.vit.num_prefix_tokens :, :]
        if self.vit.dynamic_img_size:
            x = x.transpose(1, 2)  # NLC -> NCL
            total_size = torch.numel(x)
            batch_size = x.size(0)
            num_channels = x.size(1)
            grid_size = int(math.sqrt(total_size / (batch_size * num_channels)))
            x = x.view(
                x.size(0),
                x.size(1),
                grid_size,
                grid_size,
            )  # NCL -> NCHW

            # NCHW -> NHWC
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.vit.pos_embed,
                (H, W),
                num_prefix_tokens=(
                    0 if self.vit.no_embed_class else self.vit.num_prefix_tokens
                ),
                antialias=self.antialias,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.vit.pos_embed

        if self.vit.no_embed_class:
            x = x + pos_embed
            if self.vit.num_prefix_tokens:
                x = torch.cat((x_prefix, x), dim=1)
        else:
            if self.vit.num_prefix_tokens:
                x = torch.cat((x_prefix, x), dim=1)
            x = x + pos_embed
        out: Tensor = self.vit.pos_drop(x)
        return out

    def _initialize_weights(self) -> None:
        # Initialize the patch embedding layer like a linear layer instead of conv
        # layer.
        w = self.vit.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # Initialize the class token.
        if self.vit.has_class_token:
            torch.nn.init.normal_(self.vit.cls_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(init_weights)


def init_weights(module: Module) -> None:
    if isinstance(module, Linear):
        nn.init.xavier_uniform_(module.weight)
        if isinstance(module, Linear) and module.bias is not None:
            nn.init.constant_(module.bias, 0)
    elif isinstance(module, LayerNorm):
        nn.init.constant_(module.bias, 0)
        nn.init.constant_(module.weight, 1.0)
        
from functools import partial
from typing import Callable, Optional

import torch
import torch.nn as nn
from timm.models.vision_transformer import Block
from torch import Tensor
from torch.nn import LayerNorm, Module, Parameter, Sequential

from lightly.models import utils
from lightly.models.modules.masked_vision_transformer_timm import init_weights

class MAEDecoderTIMM(Module):
    """Decoder for the Masked Autoencoder model [0].

    Decodes encoded patches and predicts pixel values for every patch.
    Code inspired by [1].

    - [0]: Masked Autoencoder, 2021, https://arxiv.org/abs/2111.06377
    - [1]: https://github.com/facebookresearch/mae

    Attributes:
        num_patches:
            Number of patches.
        patch_size:
            Patch size.
        in_chans:
            Number of image input channels.
        embed_dim:
            Embedding dimension of the encoder.
        decoder_embed_dim:
            Embedding dimension of the decoder.
        decoder_depth:
            Depth of transformer.
        decoder_num_heads:
            Number of attention heads.
        mlp_ratio:
            Ratio of mlp hidden dim to embedding dim.
        proj_drop_rate:
            Percentage of elements set to zero after the MLP in the transformer.
        attn_drop_rate:
            Percentage of elements set to zero after the attention head.
        norm_layer:
            Normalization layer.
        initialize_weights:
            Flag that determines if weights should be initialized.
        mask_token:
            The mask token.

    """

    def __init__(
        self,
        num_patches: int,
        patch_size: int,
        in_chans: int = 3,
        embed_dim: int = 1024,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        proj_drop_rate: float = 0.0,
        attn_drop_rate: float = 0.0,
        norm_layer: Callable[..., nn.Module] = partial(LayerNorm, eps=1e-6),
        initialize_weights: bool = True,
        mask_token: Optional[Parameter] = None,
    ):
        """Initializes the MAEDecoderTIMM with the specified parameters."""

        super().__init__()

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)
        self.mask_token = (
            nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            if mask_token is None
            else mask_token
        )

        # Positional encoding of the decoder
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim), requires_grad=False
        )  # fixed sin-cos embedding

        self.decoder_blocks = Sequential(
            *[
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=True,
                    norm_layer=norm_layer,
                    proj_drop=proj_drop_rate,
                    attn_drop=attn_drop_rate,
                )
                for i in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim, patch_size**2 * in_chans, bias=True
        )  # decoder to patch

        if initialize_weights:
            self._initialize_weights()

    def forward(self, input: Tensor) -> Tensor:
        """Returns predicted pixel values from encoded tokens.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, embed_input_dim).

        Returns:
            Tensor with shape (batch_size, seq_length, out_dim).
        """

        out = self.embed(input)
        out = self.decode(out)
        return self.predict(out)

    def embed(self, input: Tensor) -> Tensor:
        """Embeds encoded input tokens into decoder token dimension.

        This is a single linear layer that changes the token dimension from
        embed_input_dim to hidden_dim.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, embed_input_dim)
                containing the encoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, hidden_dim) containing
            the embedded tokens.

        """
        out: Tensor = self.decoder_embed(input)
        return out

    def decode(self, input: Tensor) -> Tensor:
        """Forward pass through the decoder transformer.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, hidden_dim) containing
                the encoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, hidden_dim) containing
            the decoded tokens.

        """
        output: Tensor = input + self.decoder_pos_embed
        output = self.decoder_blocks(output)
        output = self.decoder_norm(output)
        return output

    def predict(self, input: Tensor) -> Tensor:
        """Predicts pixel values from decoded tokens.

        Args:
            input:
                Tensor with shape (batch_size, seq_length, hidden_dim) containing
                the decoded tokens.

        Returns:
            Tensor with shape (batch_size, seq_length, out_dim) containing
            predictions for each token.

        """
        out: Tensor = self.decoder_pred(input)
        return out

    def _initialize_weights(self) -> None:
        """Initializes weights for the decoder components."""

        torch.nn.init.normal_(self.mask_token, std=0.02)
        utils.initialize_2d_sine_cosine_positional_embedding(
            pos_embedding=self.decoder_pos_embed, has_class_token=True
        )
        self.apply(init_weights)