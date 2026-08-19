"""Building blocks for the segmentation network."""

from __future__ import annotations

import torch
from torch import nn


class ConvBlock(nn.Module):
    """Two 3x3 convolutions with group normalisation and a residual shortcut.

    Group normalisation is used in preference to batch normalisation because
    filament crops are large, which forces small batches; batch statistics
    estimated from two or four samples are too noisy to be useful.
    """

    def __init__(self, in_channels: int, out_channels: int, groups: int = 8) -> None:
        super().__init__()
        norm_groups = _pick_groups(out_channels, groups)
        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(norm_groups, out_channels),
        )
        self.shortcut = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.body(x) + self.shortcut(x))


class Down(nn.Module):
    """Halve the spatial resolution, then apply a :class:`ConvBlock`."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.block = ConvBlock(in_channels, out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(self.pool(x))


class Up(nn.Module):
    """Upsample, concatenate the skip connection, then apply a :class:`ConvBlock`.

    Bilinear upsampling followed by a 1x1 projection is preferred to a transposed
    convolution because it does not produce the chequerboard artefacts that a
    thin-structure model would happily mistake for barbs.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.block = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = nn.functional.interpolate(
            x, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        x = self.reduce(x)
        return self.block(torch.cat([x, skip], dim=1))


def _pick_groups(channels: int, preferred: int) -> int:
    """Largest divisor of ``channels`` no greater than ``preferred``."""
    groups = min(preferred, channels)
    while groups > 1 and channels % groups != 0:
        groups -= 1
    return max(1, groups)
