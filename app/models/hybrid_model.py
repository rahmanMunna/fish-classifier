"""
Hybrid model: DFCViT-4C (MobileNetV2 + CBAM x2 + Lightweight Transformer)

Copied verbatim from the training notebook / original Gradio app.py so
that hybrid.pt's state_dict loads with zero key mismatch. DO NOT rename
any layer/module here — the checkpoint was saved against these exact
attribute names (backbone, proj_conv, proj_bn, proj_act, cbam1, cbam2,
tokenizer, transformer_blocks, encoder_norm, classifier).
"""

import torch
import torch.nn as nn
from torchvision import models

# ─────────────────────────────────────────────────────────────────────────
# Hyperparameters (must match training)
# ─────────────────────────────────────────────────────────────────────────
EMBED_DIM = 256
NUM_TRANSFORMER_BLOCKS = 2
NUM_ATTN_HEADS = 8
FFN_EXPANSION_RATIO = 2.0
DROP_PATH_MAX = 0.10
CLASSIFIER_DROPOUT = 0.3
BACKBONE_FREEZE_EPOCHS = 8


class ChannelAttention(nn.Module):
    """CBAM channel-attention sub-module: avg-pool + max-pool through a shared MLP."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        hidden = max(channels // reduction, 8)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        attn = self.sigmoid(avg_out + max_out)
        return x * attn


class SpatialAttention(nn.Module):
    """CBAM spatial-attention sub-module: pooled feature maps -> spatial attention map."""

    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        pooled = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(pooled))
        return x * attn


class CBAM(nn.Module):
    """Convolutional Block Attention Module: Channel Attention -> Spatial Attention."""

    def __init__(self, channels: int, reduction: int = 16, kernel_size: int = 7):
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


class PatchTokenizer(nn.Module):
    """Flattens a (B, C, H, W) feature map into spatial tokens, prepends a
    learnable [CLS] token, then adds a learnable positional embedding."""

    def __init__(self, embed_dim: int = 256, num_patches: int = 49):
        super().__init__()
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)
        tokens = tokens + self.pos_embed
        return tokens


class DropPath(nn.Module):
    """Stochastic Depth per sample (DropPath)."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class TransformerEncoderBlock(nn.Module):
    """Lightweight Transformer encoder block: Pre-LN + MHSA + GELU-FFN + DropPath."""

    def __init__(self, dim: int = 256, num_heads: int = 8, mlp_ratio: float = 2.0,
                 drop_path: float = 0.0, attn_dropout: float = 0.0, proj_dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_dropout, batch_first=True)
        self.drop_path1 = DropPath(drop_path)

        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(proj_dropout),
        )
        self.drop_path2 = DropPath(drop_path)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        x = x + self.drop_path1(attn_out)
        x = x + self.drop_path2(self.mlp(self.norm2(x)))
        return x


class DFCViT4C(nn.Module):
    """DFCViT-4C — Lightweight CNN + CBAM + Transformer Hybrid Model."""

    def __init__(self, num_classes: int, embed_dim: int = EMBED_DIM,
                 num_heads: int = NUM_ATTN_HEADS,
                 num_transformer_blocks: int = NUM_TRANSFORMER_BLOCKS,
                 mlp_ratio: float = FFN_EXPANSION_RATIO,
                 dropout: float = CLASSIFIER_DROPOUT,
                 drop_path_max: float = DROP_PATH_MAX,
                 backbone_freeze_epochs: int = BACKBONE_FREEZE_EPOCHS):
        super().__init__()
        self.backbone_freeze_epochs = backbone_freeze_epochs
        self.embed_dim = embed_dim

        try:
            mobilenet = models.mobilenet_v2(weights=None)
        except Exception:
            mobilenet = models.mobilenet_v2(pretrained=False)

        # Drop the final 320->1280 1x1 conv so the backbone emits (320, 7, 7).
        self.backbone = nn.Sequential(*list(mobilenet.features.children())[:-1])
        self.backbone_out_channels = 320

        self.proj_conv = nn.Conv2d(self.backbone_out_channels, embed_dim, kernel_size=1, bias=False)
        self.proj_bn = nn.BatchNorm2d(embed_dim)
        self.proj_act = nn.GELU()

        self.cbam1 = CBAM(embed_dim)
        self.cbam2 = CBAM(embed_dim)

        self.tokenizer = PatchTokenizer(embed_dim=embed_dim, num_patches=7 * 7)

        drop_path_rates = torch.linspace(0, drop_path_max, num_transformer_blocks).tolist()
        self.transformer_blocks = nn.ModuleList([
            TransformerEncoderBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, drop_path=dp)
            for dp in drop_path_rates
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)

        self.classifier = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True

    def get_param_counts(self) -> dict:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {"total": total, "trainable": trainable, "frozen": total - trainable}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.backbone(x)
        feat = self.proj_act(self.proj_bn(self.proj_conv(feat)))
        feat = self.cbam1(feat)
        feat = self.cbam2(feat)

        tokens = self.tokenizer(feat)
        for block in self.transformer_blocks:
            tokens = block(tokens)
        tokens = self.encoder_norm(tokens)

        cls_repr = tokens[:, 0]
        return self.classifier(cls_repr)