"""Edge-guided self-attention.

A filament is defined as much by its outline as by its interior: the barbs that
the challenge scores most heavily are thin threads whose only signature is a
pair of nearby edges.  Plain self-attention at a U-Net bottleneck has no notion
of that; it mixes tokens by feature similarity alone, and a barb's tokens look
much like the quiet Sun around them.

The module below extracts an explicit edge map from the input image with a
learnable filter bank (initialised to Sobel and Laplacian kernels, so it starts
out as a real edge detector and refines from there), then uses that map to
modulate the attention Queries and Keys.  Tokens that sit on a strong edge
therefore attend differently from tokens in a smooth region, and the network
gets an inductive bias towards boundary structure without being told where any
particular filament is.

The edge map also removes the need for a learned positional encoding: it varies
across the image in a way that is tied to the content, which is what the
positional encoding was providing in the first place.
"""

from __future__ import annotations

import math

import torch
from torch import nn


def _edge_kernels() -> torch.Tensor:
    """A small bank of classical edge kernels, used to initialise the extractor."""
    sobel_x = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
    sobel_y = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
    diagonal_a = [[0.0, 1.0, 2.0], [-1.0, 0.0, 1.0], [-2.0, -1.0, 0.0]]
    diagonal_b = [[2.0, 1.0, 0.0], [1.0, 0.0, -1.0], [0.0, -1.0, -2.0]]
    laplacian = [[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]
    # A ridge filter: filaments are dark ridges, not step edges, so an explicit
    # second-derivative kernel gives the bank something tuned to their shape.
    ridge = [[-1.0, 2.0, -1.0], [-1.0, 2.0, -1.0], [-1.0, 2.0, -1.0]]
    kernels = [sobel_x, sobel_y, diagonal_a, diagonal_b, laplacian, ridge]
    return torch.tensor(kernels, dtype=torch.float32).unsqueeze(1) / 4.0


class LearnableEdgeMap(nn.Module):
    """Produce a multi-channel edge embedding from the raw input image.

    Args:
        in_channels: Channels in the model input.
        out_channels: Width of the edge embedding.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 32) -> None:
        super().__init__()
        bank = _edge_kernels()
        n_kernels = bank.shape[0]
        self.filters = nn.Conv2d(in_channels, n_kernels, 3, padding=1, bias=False)
        with torch.no_grad():
            weight = bank.repeat(1, in_channels, 1, 1) / max(in_channels, 1)
            self.filters.weight.copy_(weight)

        self.project = nn.Sequential(
            nn.Conv2d(n_kernels, out_channels, 1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(min(8, out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Rectifying the responses means the embedding measures edge strength
        # rather than edge polarity, which is what we want to steer attention.
        return self.project(torch.abs(self.filters(x)))


class EdgeGuidedAttention(nn.Module):
    """Multi-head self-attention whose Queries and Keys are modulated by edges.

    Values are deliberately left untouched: the edge map should decide *what
    attends to what*, not overwrite the semantic content being aggregated.

    Args:
        channels: Width of the feature map being attended over.
        edge_channels: Width of the edge embedding.
        n_heads: Number of attention heads.
        dropout: Dropout applied to the attention output.
    """

    def __init__(
        self,
        channels: int,
        edge_channels: int = 32,
        n_heads: int = 8,
        dropout: float = 0.0,
        use_edge: bool = True,
    ) -> None:
        super().__init__()
        self.use_edge = bool(use_edge)
        while channels % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.norm = nn.GroupNorm(min(8, channels), channels)
        self.to_q = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_k = nn.Conv2d(channels, channels, 1, bias=False)
        self.to_v = nn.Conv2d(channels, channels, 1, bias=False)

        # The linear maps that carry edge information into Q and K. With
        # ``use_edge`` off the block degenerates to ordinary self-attention,
        # which is the ablation baseline.
        if self.use_edge:
            self.edge_to_q = nn.Conv2d(edge_channels, channels, 1, bias=False)
            self.edge_to_k = nn.Conv2d(edge_channels, channels, 1, bias=False)
            # Start at zero so the block begins as ordinary self-attention and
            # learns how much edge guidance to admit. This keeps early training
            # stable no matter how the edge bank is scaled.
            nn.init.zeros_(self.edge_to_q.weight)
            nn.init.zeros_(self.edge_to_k.weight)

        self.project = nn.Conv2d(channels, channels, 1)
        self.dropout = nn.Dropout(dropout)

        self.mlp = nn.Sequential(
            nn.GroupNorm(min(8, channels), channels),
            nn.Conv2d(channels, channels * 2, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels * 2, channels, 1),
        )

    def forward(self, x: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
        batch, channels, height, width = x.shape
        if self.use_edge and edge.shape[-2:] != (height, width):
            edge = nn.functional.interpolate(
                edge, size=(height, width), mode="bilinear", align_corners=False
            )

        normed = self.norm(x)
        query = self.to_q(normed)
        key = self.to_k(normed)
        if self.use_edge:
            query = query + self.edge_to_q(edge)
            key = key + self.edge_to_k(edge)
        value = self.to_v(normed)

        def heads(tensor: torch.Tensor) -> torch.Tensor:
            return tensor.reshape(batch, self.n_heads, self.head_dim, height * width).transpose(2, 3)

        attended = nn.functional.scaled_dot_product_attention(
            heads(query), heads(key), heads(value), scale=self.scale
        )
        attended = attended.transpose(2, 3).reshape(batch, channels, height, width)

        x = x + self.dropout(self.project(attended))
        return x + self.mlp(x)
