"""Loss functions built for thin, sparse structures."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


def soft_erode(x: torch.Tensor) -> torch.Tensor:
    """Greyscale erosion by a cross-shaped structuring element."""
    vertical = -F.max_pool2d(-x, (3, 1), stride=1, padding=(1, 0))
    horizontal = -F.max_pool2d(-x, (1, 3), stride=1, padding=(0, 1))
    return torch.min(vertical, horizontal)


def soft_dilate(x: torch.Tensor) -> torch.Tensor:
    """Greyscale dilation by a 3x3 square structuring element."""
    return F.max_pool2d(x, (3, 3), stride=1, padding=1)


def soft_open(x: torch.Tensor) -> torch.Tensor:
    """Greyscale opening: erosion followed by dilation."""
    return soft_dilate(soft_erode(x))


def soft_skeleton(x: torch.Tensor, iterations: int = 10) -> torch.Tensor:
    """A differentiable approximation of the morphological skeleton."""
    opened = soft_open(x)
    skeleton = F.relu(x - opened)
    for _ in range(iterations):
        x = soft_erode(x)
        opened = soft_open(x)
        delta = F.relu(x - opened)
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def cl_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    iterations: int = 10,
    smooth: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Centreline Dice loss, optionally restricted to a validity mask."""
    probability = torch.sigmoid(logits)
    if mask is not None:
        probability = probability * mask
        target = target * mask

    skeleton_pred = soft_skeleton(probability, iterations)
    skeleton_true = soft_skeleton(target, iterations)
    if mask is not None:
        skeleton_pred = skeleton_pred * mask
        skeleton_true = skeleton_true * mask

    dims = tuple(range(1, target.dim()))
    precision = (torch.sum(skeleton_pred * target, dim=dims) + smooth) / (
        torch.sum(skeleton_pred, dim=dims) + smooth
    )
    sensitivity = (torch.sum(skeleton_true * probability, dim=dims) + smooth) / (
        torch.sum(skeleton_true, dim=dims) + smooth
    )
    cl_dice = 2.0 * precision * sensitivity / (precision + sensitivity + 1e-8)
    return (1.0 - cl_dice).mean()


def dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Soft Sørensen-Dice loss for binary segmentation.

    This is intentionally simpler than Tversky and clDice. It exists so the
    large-context ResNet U-Net experiment can use the widely established
    ``BCE + Dice`` objective as a clean control instead of changing architecture,
    context and loss complexity at the same time.
    """
    probability = torch.sigmoid(logits)
    if mask is not None:
        probability = probability * mask
        target = target * mask

    dims = tuple(range(1, target.dim()))
    intersection = torch.sum(probability * target, dim=dims)
    denominator = torch.sum(probability, dim=dims) + torch.sum(target, dim=dims)
    score = (2.0 * intersection + smooth) / (denominator + smooth)
    return (1.0 - score).mean()


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tversky loss, a Dice generalisation with asymmetric error weighting."""
    probability = torch.sigmoid(logits)
    if mask is not None:
        probability = probability * mask
        target = target * mask

    dims = tuple(range(1, target.dim()))
    true_positive = torch.sum(probability * target, dim=dims)
    false_positive = torch.sum(probability * (1.0 - target), dim=dims)
    false_negative = torch.sum((1.0 - probability) * target, dim=dims)
    score = (true_positive + smooth) / (
        true_positive + alpha * false_positive + beta * false_negative + smooth
    )
    return (1.0 - score).mean()


def weighted_bce(
    logits: torch.Tensor,
    target: torch.Tensor,
    weight: torch.Tensor | None = None,
    mask: torch.Tensor | None = None,
    pos_weight: float | None = None,
) -> torch.Tensor:
    """Binary cross-entropy with per-pixel weights and an optional validity mask."""
    pos = (
        torch.tensor(pos_weight, device=logits.device, dtype=logits.dtype)
        if pos_weight is not None
        else None
    )
    loss = F.binary_cross_entropy_with_logits(
        logits, target, reduction="none", pos_weight=pos
    )
    if weight is not None:
        loss = loss * weight
    if mask is not None:
        loss = loss * mask
        denominator = mask.sum().clamp_min(1.0)
        return loss.sum() / denominator
    return loss.mean()


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Focal loss: down-weights easy quiet-Sun pixels."""
    probability = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    p_t = probability * target + (1.0 - probability) * (1.0 - target)
    alpha_t = alpha * target + (1.0 - alpha) * (1.0 - target)
    loss = alpha_t * (1.0 - p_t).pow(gamma) * ce
    if mask is not None:
        loss = loss * mask
        return loss.sum() / mask.sum().clamp_min(1.0)
    return loss.mean()


@dataclass
class LossWeights:
    """Relative weights and switches used by :class:`FilamentLoss`.

    ``dice`` defaults to zero to preserve every historical FilaNet run exactly.
    ``use_distance_weight`` likewise defaults to the old behaviour. A plain
    ResNet U-Net control can therefore request BCE+Dice and disable handcrafted
    distance weighting without changing the legacy objective.
    """

    bce: float = 1.0
    dice: float = 0.0
    tversky: float = 1.0
    cl_dice: float = 1.0
    focal: float = 0.5
    spine: float = 0.5
    boundary: float = 0.5
    deep: float = 0.4
    use_distance_weight: bool = True


class FilamentLoss(nn.Module):
    """The configurable segmentation objective used to train FilaNet variants."""

    def __init__(
        self,
        weights: LossWeights | None = None,
        tversky_alpha: float = 0.3,
        tversky_beta: float = 0.7,
        cl_dice_iterations: int = 10,
        pos_weight: float | None = 4.0,
        cl_dice_warmup: int = 500,
    ) -> None:
        super().__init__()
        self.weights = weights or LossWeights()
        self.tversky_alpha = tversky_alpha
        self.tversky_beta = tversky_beta
        self.cl_dice_iterations = cl_dice_iterations
        self.pos_weight = pos_weight
        self.cl_dice_warmup = max(0, int(cl_dice_warmup))
        self.register_buffer("_step", torch.zeros((), dtype=torch.long))

    def _cl_dice_scale(self) -> float:
        if self.cl_dice_warmup == 0:
            return 1.0
        return float(min(1.0, float(self._step.item()) / self.cl_dice_warmup))

    def forward(
        self,
        outputs: dict[str, torch.Tensor | list[torch.Tensor]],
        batch: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        logits = outputs["mask"]
        assert isinstance(logits, torch.Tensor)
        target = batch["mask"]
        valid = batch.get("valid")
        weight = batch.get("weight") if self.weights.use_distance_weight else None

        components: dict[str, float] = {}
        total = logits.new_zeros(())

        if self.weights.bce > 0:
            term = weighted_bce(logits, target, weight, valid, self.pos_weight)
            total = total + self.weights.bce * term
            components["bce"] = float(term.detach())

        if self.weights.dice > 0:
            term = dice_loss(logits, target, mask=valid)
            total = total + self.weights.dice * term
            components["dice"] = float(term.detach())

        if self.weights.tversky > 0:
            term = tversky_loss(
                logits, target, self.tversky_alpha, self.tversky_beta, mask=valid
            )
            total = total + self.weights.tversky * term
            components["tversky"] = float(term.detach())

        if self.weights.focal > 0:
            term = focal_loss(logits, target, mask=valid)
            total = total + self.weights.focal * term
            components["focal"] = float(term.detach())

        if self.weights.cl_dice > 0:
            scale = self._cl_dice_scale()
            term = cl_dice_loss(
                logits,
                target,
                self.cl_dice_iterations,
                mask=valid,
            )
            total = total + self.weights.cl_dice * scale * term
            components["cl_dice"] = float(term.detach())

        if self.weights.spine > 0 and "spine" in outputs and "spine" in batch:
            spine_logits = outputs["spine"]
            assert isinstance(spine_logits, torch.Tensor)
            term = weighted_bce(spine_logits, batch["spine"], mask=valid)
            total = total + self.weights.spine * term
            components["spine"] = float(term.detach())

        if (
            self.weights.boundary > 0
            and "boundary" in outputs
            and "boundary" in batch
        ):
            boundary_logits = outputs["boundary"]
            assert isinstance(boundary_logits, torch.Tensor)
            term = weighted_bce(boundary_logits, batch["boundary"], mask=valid)
            total = total + self.weights.boundary * term
            components["boundary"] = float(term.detach())

        deep = outputs.get("deep")
        if self.weights.deep > 0 and isinstance(deep, list) and deep:
            deep_total = logits.new_zeros(())
            for deep_logits in deep:
                size = deep_logits.shape[-2:]
                small_target = F.adaptive_max_pool2d(target, size)
                small_valid = (
                    F.adaptive_max_pool2d(valid, size)
                    if valid is not None
                    else None
                )
                deep_total = deep_total + weighted_bce(
                    deep_logits,
                    small_target,
                    mask=small_valid,
                    pos_weight=self.pos_weight,
                )
            deep_total = deep_total / len(deep)
            total = total + self.weights.deep * deep_total
            components["deep"] = float(deep_total.detach())

        if self.training:
            self._step += 1
        components["total"] = float(total.detach())
        return total, components
