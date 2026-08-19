"""FilaNet: an edge-guided, multi-task U-Net for solar filament segmentation.

The design responds to three specific properties of the task.

**Filaments are thin.**  A U-Net's deepest features have a large receptive
field but a coarse grid, and barbs disappear at that grid.  FilaNet therefore
keeps a comparatively shallow encoder (four downsamplings by default) and puts
its capacity into a wide, well-normalised decoder that runs back out to full
resolution.

**Boundaries carry the signal.**  The bottleneck uses
:class:`~filaseg.models.edge_attention.EdgeGuidedAttention`, so long-range
mixing is steered by an explicit, learnable edge map taken from the input.

**One task is not enough supervision.**  Alongside the mask, the network
predicts the filament centreline and its boundary.  These auxiliary heads cost
almost nothing at inference (they can simply be ignored) but they force the
shared decoder to represent the skeleton and outline explicitly, which is what
keeps barbs alive.  Deep supervision on the mask at several decoder scales
gives the encoder a short gradient path.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .blocks import ConvBlock, Down, Up
from .edge_attention import EdgeGuidedAttention, LearnableEdgeMap


@dataclass
class FilaNetConfig:
    """Architecture hyper-parameters."""

    in_channels: int = 2
    base_width: int = 32
    depth: int = 4
    edge_channels: int = 32
    n_heads: int = 8
    dropout: float = 0.0
    deep_supervision: bool = True
    aux_heads: bool = True
    edge_attention: bool = True
    """Steer bottleneck attention with the learnable edge map. Off = plain MHSA."""
    channel_multipliers: tuple[int, ...] = field(default_factory=lambda: (1, 2, 4, 8, 16))


class FilaNet(nn.Module):
    """Edge-guided multi-task U-Net.

    Args:
        config: Architecture settings; the defaults are the ones used for the
            reported results.

    The forward pass returns a dictionary so that callers can take just the
    mask and ignore everything else:

    * ``mask`` -- filament logits at full resolution;
    * ``spine`` -- centreline logits, if auxiliary heads are enabled;
    * ``boundary`` -- outline logits, if auxiliary heads are enabled;
    * ``deep`` -- a list of lower-resolution mask logits for deep supervision.
    """

    def __init__(self, config: FilaNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or FilaNetConfig()
        cfg = self.config

        widths = [cfg.base_width * m for m in cfg.channel_multipliers[: cfg.depth + 1]]
        self.widths = widths

        self.stem = ConvBlock(cfg.in_channels, widths[0])
        self.downs = nn.ModuleList(
            [Down(widths[i], widths[i + 1]) for i in range(cfg.depth)]
        )

        self.edge_map = LearnableEdgeMap(cfg.in_channels, cfg.edge_channels)
        self.attention = EdgeGuidedAttention(
            widths[-1], cfg.edge_channels, cfg.n_heads, cfg.dropout,
            use_edge=cfg.edge_attention,
        )

        self.ups = nn.ModuleList(
            [
                Up(widths[i + 1], widths[i], widths[i])
                for i in reversed(range(cfg.depth))
            ]
        )

        self.mask_head = nn.Conv2d(widths[0], 1, 1)
        if cfg.aux_heads:
            self.spine_head = nn.Conv2d(widths[0], 1, 1)
            self.boundary_head = nn.Conv2d(widths[0], 1, 1)
        if cfg.deep_supervision:
            # One auxiliary mask head per decoder stage below full resolution.
            self.deep_heads = nn.ModuleList(
                [nn.Conv2d(widths[i], 1, 1) for i in reversed(range(1, cfg.depth))]
            )

        self.apply(_init_weights)
        # Bias the mask head towards "background". Filaments are well under one
        # per cent of pixels, so starting from a neutral bias wastes the first
        # epochs undoing a hugely over-confident positive prediction.
        with torch.no_grad():
            nn.init.constant_(self.mask_head.bias, -4.0)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        edge = self.edge_map(x)

        skips: list[torch.Tensor] = []
        feature = self.stem(x)
        for down in self.downs:
            skips.append(feature)
            feature = down(feature)

        feature = self.attention(feature, edge)

        deep_outputs: list[torch.Tensor] = []
        for index, up in enumerate(self.ups):
            feature = up(feature, skips[-(index + 1)])
            if self.config.deep_supervision and index < len(self.ups) - 1:
                deep_outputs.append(self.deep_heads[index](feature))

        outputs: dict[str, torch.Tensor | list[torch.Tensor]] = {
            "mask": self.mask_head(feature)
        }
        if self.config.aux_heads:
            outputs["spine"] = self.spine_head(feature)
            outputs["boundary"] = self.boundary_head(feature)
        if self.config.deep_supervision:
            outputs["deep"] = deep_outputs
        return outputs

    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Return filament probabilities at full resolution."""
        self.eval()
        return torch.sigmoid(self.forward(x)["mask"])  # type: ignore[arg-type]

    def n_parameters(self, trainable_only: bool = True) -> int:
        """Count parameters, for reporting model size."""
        return sum(
            p.numel()
            for p in self.parameters()
            if p.requires_grad or not trainable_only
        )


def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        # The edge bank is deliberately pre-initialised; do not overwrite it.
        if getattr(module, "_keep_init", False):
            return
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_model(config: FilaNetConfig | dict | None = None) -> FilaNet:
    """Construct a :class:`FilaNet` from a config object or a plain dictionary."""
    if isinstance(config, dict):
        known = {f for f in FilaNetConfig.__dataclass_fields__}
        config = FilaNetConfig(**{k: v for k, v in config.items() if k in known})
    return FilaNet(config)
