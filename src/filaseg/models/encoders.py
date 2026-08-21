"""Pretrained encoders for FilaNet.

MAGFiLO gives roughly seven hundred distinct observations. That is a small
number to learn general image features from, and a segmentation network trained
from scratch on it spends much of its capacity rediscovering edges, textures and
gradients that ImageNet models already represent well. Starting from pretrained
weights is the single largest change available here, and it costs nothing at
inference: the decoder and the edge-guided bottleneck are unchanged.

Two details matter for solar images specifically.

**The input is not RGB.** FilaNet takes a flattened intensity channel and a
geometry channel. The pretrained first convolution expects three channels, so
its weights are averaged across the colour axis and repeated, which preserves
the learned spatial filters while accepting a different channel count. Scaling
by ``3 / in_channels`` keeps the activation magnitudes the encoder was tuned
for.

**Filaments are thin.** A standard ResNet stem downsamples by four immediately,
through a stride-2 convolution followed by max pooling, and structures a few
pixels wide do not survive that. The stem's pooling is therefore removed, so the
encoder reaches the same depth with one fewer early downsampling and the finest
skip connection is at half resolution rather than a quarter.
"""

from __future__ import annotations

import torch
from torch import nn

#: Encoders that can be requested by name, with the channel widths each stage
#: emits once the stem pooling has been removed.
AVAILABLE = {
    "resnet18": (64, 64, 128, 256, 512),
    "resnet34": (64, 64, 128, 256, 512),
    "resnet50": (64, 256, 512, 1024, 2048),
}


def adapt_first_convolution(conv: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """Rebuild a pretrained first convolution for a different channel count.

    The pretrained kernel is averaged over its input channels and repeated, so
    every new channel starts from the mean RGB filter rather than from noise.
    The result is rescaled by ``3 / in_channels`` so the layer's output keeps
    roughly the magnitude the rest of the network was trained against.
    """
    adapted = nn.Conv2d(
        in_channels,
        conv.out_channels,
        kernel_size=conv.kernel_size,  # type: ignore[arg-type]
        stride=conv.stride,  # type: ignore[arg-type]
        padding=conv.padding,  # type: ignore[arg-type]
        bias=conv.bias is not None,
    )
    with torch.no_grad():
        mean_kernel = conv.weight.mean(dim=1, keepdim=True)
        scale = conv.weight.shape[1] / max(in_channels, 1)
        adapted.weight.copy_(mean_kernel.repeat(1, in_channels, 1, 1) * scale)
        if conv.bias is not None and adapted.bias is not None:
            adapted.bias.copy_(conv.bias)
    return adapted


class ResNetEncoder(nn.Module):
    """A torchvision ResNet as a U-Net encoder, adapted for thin structures.

    Args:
        name: One of :data:`AVAILABLE`.
        in_channels: Channels in the model input.
        pretrained: Load ImageNet weights. Turn off for a controlled comparison
            against training from scratch.
        keep_stem_pool: Keep the ResNet stem's max pooling. Off by default,
            which halves the early downsampling so filament barbs survive to
            the first skip connection.

    The forward pass returns five feature maps, from finest to coarsest, for the
    decoder to consume as skip connections.
    """

    def __init__(
        self,
        name: str = "resnet34",
        in_channels: int = 2,
        pretrained: bool = True,
        keep_stem_pool: bool = False,
    ) -> None:
        super().__init__()
        if name not in AVAILABLE:
            raise ValueError(
                f"unknown encoder {name!r}; choose from {sorted(AVAILABLE)}"
            )
        try:
            from torchvision import models
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise ImportError(
                "Pretrained encoders need torchvision. Install it with "
                "'pip install torchvision', or set encoder='scratch'."
            ) from exc

        weights = "IMAGENET1K_V1" if pretrained else None
        backbone = getattr(models, name)(weights=weights)

        self.widths = AVAILABLE[name]
        self.name = name
        self.keep_stem_pool = keep_stem_pool

        self.stem = nn.Sequential(
            adapt_first_convolution(backbone.conv1, in_channels),
            backbone.bn1,
            backbone.relu,
        )
        self.pool = backbone.maxpool if keep_stem_pool else nn.Identity()
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

    @property
    def downsampling(self) -> int:
        """Total factor by which the coarsest feature map is reduced."""
        return 32 if self.keep_stem_pool else 16

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return the five encoder feature maps, finest first."""
        stem = self.stem(x)
        first = self.layer1(self.pool(stem))
        second = self.layer2(first)
        third = self.layer3(second)
        fourth = self.layer4(third)
        return [stem, first, second, third, fourth]
