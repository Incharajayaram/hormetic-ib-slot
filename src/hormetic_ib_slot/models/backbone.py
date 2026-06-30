"""
CNN feature extractors and positional embeddings for slot attention.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """
    4-layer CNN for 64x64 input images.
    Channels: 3 -> 32 -> 32 -> 64 -> 64
    Each layer: Conv2d(3x3, stride=1, pad=1) + GroupNorm(8) + ReLU
    Output shape: (B, 64, H, W) — spatial dimensions unchanged.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualBlock(nn.Module):
    """Conv -> GN -> ReLU -> Conv -> GN + skip connection."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.gn1(self.conv1(x)), inplace=True)
        out = self.gn2(self.conv2(out))
        return F.relu(out + residual, inplace=True)


class ResNetBackbone(nn.Module):
    """
    4-block residual network for feature extraction.
    Input: (B, 3, H, W)
    Output: (B, 64, H, W)
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 64) -> None:
        super().__init__()
        self.out_channels = out_channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.blocks(self.stem(x))


class PositionEmbedding(nn.Module):
    """
    Adds 2D sinusoidal positional embeddings to a (B, C, H, W) feature map.

    The spatial grid uses torch.linspace in [-1, 1]. The first C//2 channels
    encode the x-axis frequencies, the remaining C//2 encode the y-axis.
    Frequencies follow the standard 1/10000^(2i/d) pattern.
    """

    def __init__(self, channels: int, temperature: float = 10000.0) -> None:
        super().__init__()
        assert channels % 2 == 0, "channels must be even for 2D positional embedding"
        self.channels = channels
        self.temperature = temperature

    def _make_embedding(
        self, height: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        half_c = self.channels // 2
        # Frequency bands for each half
        dim_t = torch.arange(half_c, device=device, dtype=dtype)
        # Use //2 so consecutive pairs share a frequency
        freq = self.temperature ** (2 * (dim_t // 2) / half_c)

        # Spatial grids in [-1, 1]
        y_grid = torch.linspace(-1, 1, height, device=device, dtype=dtype)  # (H,)
        x_grid = torch.linspace(-1, 1, width, device=device, dtype=dtype)   # (W,)

        # Outer product to get 2D positions (H, W, half_c)
        y_embed = y_grid[:, None] / freq[None, :]   # (H, half_c)
        x_embed = x_grid[:, None] / freq[None, :]   # (W, half_c)

        # Apply sin/cos to alternate indices
        y_sin = torch.sin(y_embed[:, 0::2])
        y_cos = torch.cos(y_embed[:, 1::2])
        x_sin = torch.sin(x_embed[:, 0::2])
        x_cos = torch.cos(x_embed[:, 1::2])

        # Interleave sin and cos: (H, half_c)
        y_enc = torch.zeros_like(y_embed)
        y_enc[:, 0::2] = y_sin
        y_enc[:, 1::2] = y_cos
        x_enc = torch.zeros_like(x_embed)
        x_enc[:, 0::2] = x_sin
        x_enc[:, 1::2] = x_cos

        # Broadcast to (H, W, channels)
        y_enc = y_enc[:, None, :].expand(height, width, half_c)  # (H, W, half_c)
        x_enc = x_enc[None, :, :].expand(height, width, half_c)  # (H, W, half_c)
        pos = torch.cat([y_enc, x_enc], dim=-1)                   # (H, W, channels)
        # -> (channels, H, W)
        return pos.permute(2, 0, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, H, W)
        Returns:
            (B, C, H, W) with positional embedding added.
        """
        B, C, H, W = x.shape
        assert C == self.channels, f"Expected {self.channels} channels, got {C}"
        emb = self._make_embedding(H, W, x.device, x.dtype)  # (C, H, W)
        return x + emb.unsqueeze(0)
