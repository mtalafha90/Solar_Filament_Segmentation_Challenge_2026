"""FilaNet: an edge-guided, multi-task U-Net for solar filament segmentation.

The design responds to three specific properties of the task.

**Filaments are thin.** A U-Net's deepest features have a large receptive field
but a coarse grid, and barbs disappear at that grid. FilaNet therefore keeps a
comparatively shallow encoder and puts its capacity into a wide decoder that
returns to full resolution.

**Boundaries carry the signal.** In the full FilaNet configuration the
bottleneck uses :class:`~filaseg.models.edge_attention.EdgeGuidedAttention`, so
long-range mixing is steered by an explicit, learnable edge map taken from the
input.

**One task is not enough supervision.** Alongside the mask, the full model can
predict the filament centreline and boundary and can use deep supervision.

For controlled experiments the same class also supports ``bottleneck_attention
=False``. With a pretrained ResNet encoder, auxiliary heads disabled and deep
supervision disabled, this is a conventional ResNet U-Net. That mode is useful
for 1024-pixel native-resolution crops: it removes the quadratic self-attention
cost and cleanly tests whether spatial context and ImageNet pretraining, rather
than architectural complexity, are the main limitation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import nn

from .blocks import ConvBlock, Down, Up
from .edge_attention import EdgeGuidedAttention, LearnableEdgeMap
from .encoders import AVAILABLE as PRETRAINED_ENCODERS
from .encoders import ResNetEncoder


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
    """Steer bottleneck attention with the learnable edge map."""
    bottleneck_attention: bool = True
    """Use self-attention at the bottleneck at all.

    ``False`` bypasses both the learnable edge map and self-attention and turns
    the encoder/decoder into a plain U-Net. This is intentionally separate from
    ``edge_attention``: setting only ``edge_attention=False`` retains ordinary
    multi-head self-attention, while setting ``bottleneck_attention=False``
    removes the quadratic block completely.
    """
    encoder: str = "scratch"
    """Encoder to use: ``"scratch"`` or a torchvision ResNet name."""
    pretrained: bool = True
    """Load ImageNet weights for the chosen pretrained encoder."""
    channel_multipliers: tuple[int, ...] = field(
        default_factory=lambda: (1, 2, 4, 8, 16)
    )


class FilaNet(nn.Module):
    """Edge-guided multi-task U-Net, with an optional plain-U-Net mode."""

    def __init__(self, config: FilaNetConfig | None = None) -> None:
        super().__init__()
        self.config = config or FilaNetConfig()
        cfg = self.config

        self.pretrained_encoder = cfg.encoder != "scratch"
        if self.pretrained_encoder:
            if cfg.encoder not in PRETRAINED_ENCODERS:
                raise ValueError(
                    f"unknown encoder {cfg.encoder!r}; use 'scratch' or one of "
                    f"{sorted(PRETRAINED_ENCODERS)}"
                )
            self.encoder = ResNetEncoder(
                cfg.encoder, cfg.in_channels, cfg.pretrained
            )
            widths = list(self.encoder.widths)
        else:
            widths = [
                cfg.base_width * m
                for m in cfg.channel_multipliers[: cfg.depth + 1]
            ]
            self.stem = ConvBlock(cfg.in_channels, widths[0])
            self.downs = nn.ModuleList(
                [Down(widths[i], widths[i + 1]) for i in range(cfg.depth)]
            )
        self.widths = widths

        if cfg.bottleneck_attention:
            self.edge_map: LearnableEdgeMap | None = LearnableEdgeMap(
                cfg.in_channels, cfg.edge_channels
            )
            self.attention: EdgeGuidedAttention | None = EdgeGuidedAttention(
                widths[-1],
                cfg.edge_channels,
                cfg.n_heads,
                cfg.dropout,
                use_edge=cfg.edge_attention,
            )
        else:
            # Do not even instantiate these modules in plain-U-Net mode. Apart
            # from saving memory, this makes the ablation unambiguous: there are
            # no dormant attention parameters in the checkpoint.
            self.edge_map = None
            self.attention = None

        self.ups = nn.ModuleList(
            [
                Up(widths[i + 1], widths[i], widths[i])
                for i in reversed(range(len(widths) - 1))
            ]
        )

        # A pretrained encoder's finest feature map is at half resolution, so
        # one more interpolation/refinement is needed for a full-resolution mask.
        self.final_up = (
            ConvBlock(widths[0], widths[0])
            if self.pretrained_encoder
            else nn.Identity()
        )

        self.mask_head = nn.Conv2d(widths[0], 1, 1)
        if cfg.aux_heads:
            self.spine_head = nn.Conv2d(widths[0], 1, 1)
            self.boundary_head = nn.Conv2d(widths[0], 1, 1)
        if cfg.deep_supervision:
            self.deep_heads = nn.ModuleList(
                [
                    nn.Conv2d(widths[i], 1, 1)
                    for i in reversed(range(1, len(widths) - 1))
                ]
            )

        # Initialise newly-created layers, then restore any ImageNet encoder
        # weights. Applying Kaiming initialisation to the backbone would discard
        # exactly the pretraining the experiment is intended to test.
        if self.pretrained_encoder:
            encoder_state = {
                k: v.clone() for k, v in self.encoder.state_dict().items()
            }
            self.apply(_init_weights)
            self.encoder.load_state_dict(encoder_state)
        else:
            self.apply(_init_weights)

        with torch.no_grad():
            nn.init.constant_(self.mask_head.bias, -4.0)

    def forward(
        self, x: torch.Tensor
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if self.pretrained_encoder:
            features = self.encoder(x)
            skips, feature = features[:-1], features[-1]
        else:
            skips = []
            feature = self.stem(x)
            for down in self.downs:
                skips.append(feature)
                feature = down(feature)

        if self.attention is not None:
            assert self.edge_map is not None
            edge = self.edge_map(x)
            feature = self.attention(feature, edge)

        deep_outputs: list[torch.Tensor] = []
        for index, up in enumerate(self.ups):
            feature = up(feature, skips[-(index + 1)])
            if self.config.deep_supervision and index < len(self.ups) - 1:
                deep_outputs.append(self.deep_heads[index](feature))

        if self.pretrained_encoder:
            feature = nn.functional.interpolate(
                feature,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            feature = self.final_up(feature)

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
        if getattr(module, "_keep_init", False):
            return
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def build_model(config: FilaNetConfig | dict | None = None) -> FilaNet:
    """Construct a :class:`FilaNet` from a config object or plain dictionary."""
    if isinstance(config, dict):
        known = {f for f in FilaNetConfig.__dataclass_fields__}
        config = FilaNetConfig(
            **{k: v for k, v in config.items() if k in known}
        )
    return FilaNet(config)
