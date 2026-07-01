"""Compact HAT-inspired SR backbone + CRNN/Transformer OCR head."""
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from ocr.vocab import NUM_CLASSES

MAX_SEQ_LEN = 64  # upper bound on the OCR encoder's time dimension, for positional embeddings


@dataclass
class SRConfig:
    channels: int = 64
    n_groups: int = 2
    n_blocks: int = 4
    mlp_ratio: int = 2
    n_heads: int = 4
    scale: int = 2

    @classmethod
    def tiny(cls) -> "SRConfig":
        return cls(channels=16, n_groups=1, n_blocks=1, mlp_ratio=2, n_heads=2, scale=2)


@dataclass
class OCRConfig:
    cnn_width: int = 32
    d_model: int = 256
    n_layers: int = 3
    n_heads: int = 8
    ff_dim: int = 512

    @classmethod
    def tiny(cls) -> "OCRConfig":
        return cls(cnn_width=4, d_model=32, n_layers=1, n_heads=2, ff_dim=64)


class CAB(nn.Module):
    """Channel Attention Block: conv pair gated by a squeeze-excite branch."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.conv1 = nn.Conv2d(channels, hidden, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden, channels, 3, 1, 1)
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv2(self.act(self.conv1(x)))
        return y * self.se(y)


class HybridAttentionBlock(nn.Module):
    """Global self-attention + channel attention (parallel), HAT's hybrid idea, no window partitioning."""

    def __init__(self, channels: int, n_heads: int, mlp_ratio: int, cab_scale: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(channels)
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(channels)
        self.cab = CAB(channels)
        self.cab_scale = cab_scale
        self.norm3 = nn.LayerNorm(channels)
        hidden = channels * mlp_ratio
        self.mlp = nn.Sequential(nn.Linear(channels, hidden), nn.GELU(), nn.Linear(hidden, channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,C,H,W]
        b, c, h, w = x.shape
        tokens = x.flatten(2).transpose(1, 2)  # [B,H*W,C]

        normed = self.norm1(tokens)
        attn_out, _ = self.attn(normed, normed, normed, need_weights=False)
        tokens = tokens + attn_out

        normed_map = self.norm2(tokens).transpose(1, 2).view(b, c, h, w)
        cab_out = self.cab(normed_map).flatten(2).transpose(1, 2)
        tokens = tokens + self.cab_scale * cab_out

        tokens = tokens + self.mlp(self.norm3(tokens))
        return tokens.transpose(1, 2).view(b, c, h, w)


class RHAG(nn.Module):
    """Residual Hybrid Attention Group: a stack of HybridAttentionBlocks with a group-level residual."""

    def __init__(self, channels: int, n_blocks: int, n_heads: int, mlp_ratio: int):
        super().__init__()
        self.blocks = nn.ModuleList(
            [HybridAttentionBlock(channels, n_heads, mlp_ratio) for _ in range(n_blocks)]
        )
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for block in self.blocks:
            y = block(y)
        return x + self.conv(y)


class MiniHATSR(nn.Module):
    """Compact HAT-inspired super-resolution backbone."""

    def __init__(self, cfg: SRConfig):
        super().__init__()
        self.scale = cfg.scale
        self.head = nn.Conv2d(3, cfg.channels, 3, 1, 1)
        self.groups = nn.ModuleList(
            [RHAG(cfg.channels, cfg.n_blocks, cfg.n_heads, cfg.mlp_ratio) for _ in range(cfg.n_groups)]
        )
        self.body_conv = nn.Conv2d(cfg.channels, cfg.channels, 3, 1, 1)
        self.upsample = nn.Sequential(
            nn.Conv2d(cfg.channels, cfg.channels * cfg.scale ** 2, 3, 1, 1),
            nn.PixelShuffle(cfg.scale),
            nn.Conv2d(cfg.channels, 3, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,3,H,W] LR -> [B,3,H*scale,W*scale]
        feat = self.head(x)
        y = feat
        for group in self.groups:
            y = group(y)
        y = feat + self.body_conv(y)
        sr = self.upsample(y)
        bicubic = F.interpolate(x, scale_factor=self.scale, mode="bicubic", align_corners=False)
        return sr + bicubic


class ConvStem(nn.Module):
    """CRNN-style stem: collapses height to 1 while preserving width as the OCR time axis."""

    def __init__(self, width: int):
        super().__init__()
        c1, c2, c3, c4 = width, width * 2, width * 4, width * 8
        self.conv1 = nn.Sequential(nn.Conv2d(3, c1, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.conv2 = nn.Sequential(nn.Conv2d(c1, c2, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2))
        self.conv3 = nn.Sequential(nn.Conv2d(c2, c3, 3, 1, 1), nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(nn.Conv2d(c3, c3, 3, 1, 1), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1)))
        self.conv5 = nn.Sequential(nn.Conv2d(c3, c4, 3, 1, 1), nn.BatchNorm2d(c4), nn.ReLU(inplace=True))
        self.conv6 = nn.Sequential(
            nn.Conv2d(c4, c4, 3, 1, 1), nn.BatchNorm2d(c4), nn.ReLU(inplace=True), nn.MaxPool2d((2, 1))
        )
        self.conv7 = nn.Conv2d(c4, c4, kernel_size=(4, 1))
        self.out_channels = c4

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,3,H,W] -> [B,C,T]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)  # [B,C,1,T]
        return x.squeeze(2)


class OCRHead(nn.Module):
    """Conv stem + small Transformer encoder + CTC classifier."""

    def __init__(self, cfg: OCRConfig):
        super().__init__()
        self.stem = ConvStem(cfg.cnn_width)
        self.proj = nn.Linear(self.stem.out_channels, cfg.d_model)
        self.pos_embed = nn.Parameter(torch.zeros(1, MAX_SEQ_LEN, cfg.d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model, nhead=cfg.n_heads, dim_feedforward=cfg.ff_dim, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=cfg.n_layers)
        self.classifier = nn.Linear(cfg.d_model, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,3,H,W] -> raw logits [B,T,NUM_CLASSES]
        feat = self.stem(x).permute(0, 2, 1)  # [B,T,C]
        feat = self.proj(feat)
        t = feat.shape[1]
        feat = feat + self.pos_embed[:, :t, :]
        feat = self.encoder(feat)
        return self.classifier(feat)  # [B,T,NUM_CLASSES], pre-softmax, batch-first


class SRPlateNet(nn.Module):
    """Composes the SR backbone and OCR head: LR image in, (SR image, raw CTC logits) out.

    Both outputs are batch-first (dim 0 = batch) so this module works unmodified under
    nn.DataParallel, whose default output gathering concatenates along dim 0.
    """

    def __init__(self, sr_cfg: SRConfig, ocr_cfg: OCRConfig):
        super().__init__()
        self.sr = MiniHATSR(sr_cfg)
        self.ocr = OCRHead(ocr_cfg)

    def forward(self, lr_img: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sr_img = self.sr(lr_img)
        logits = self.ocr(sr_img)  # [B,T,NUM_CLASSES]
        return sr_img, logits
