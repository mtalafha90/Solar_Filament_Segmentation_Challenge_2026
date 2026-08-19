"""Loss functions built for thin, sparse structures.

The single most important design choice in this project is *not* the network
architecture but the loss.  Filaments occupy under one per cent of the solar
disk, and the parts the challenge cares about most -- the barbs -- are a small
fraction of that again.  Optimising plain Dice or plain cross-entropy produces
a model that draws smooth, confident blobs over filament bodies and quietly
deletes every fine thread, because doing so costs almost nothing on those
objectives.

Three ingredients fix that:

* **clDice** (centreline Dice) scores the *topology* of the prediction by
  comparing soft skeletons.  Deleting a barb breaks a branch of the skeleton,
  which clDice punishes heavily even though the pixel count barely moves.
* **Tversky** loss lets recall be weighted above precision, which counteracts
  the model's incentive to under-segment when positives are rare.
* **Per-pixel weights** concentrate the cross-entropy term on outlines and on
  locally thin regions.

Everything here operates on raw logits.
"""

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
    """A differentiable approximation of the morphological skeleton.

    This is the standard iterative thinning construction: at each step, whatever
    the opening removes is skeleton material.  It is fully differentiable, so it
    can sit inside a loss.

    Args:
        x: Probabilities in ``[0, 1]``, shape ``(B, C, H, W)``.
        iterations: Thinning steps.  This bounds the half-width of structures
            that can be reduced to a centreline, so it should comfortably exceed
            the half-width of a typical filament.
    """
    opened = soft_open(x)
    skeleton = F.relu(x - opened)
    for _ in range(iterations):
        x = soft_erode(x)
        opened = soft_open(x)
        delta = F.relu(x - opened)
        # Union without double counting where the new material overlaps.
        skeleton = skeleton + F.relu(delta - skeleton * delta)
    return skeleton


def cl_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    iterations: int = 10,
    smooth: float = 1.0,
) -> torch.Tensor:
    """Centreline Dice loss.

    Measures two things and combines them harmonically:

    * *topology precision* -- how much of the predicted skeleton lies inside the
      true mask (penalises spurious threads);
    * *topology sensitivity* -- how much of the true skeleton lies inside the
      predicted mask (penalises deleted barbs).
    """
    probability = torch.sigmoid(logits)
    skeleton_pred = soft_skeleton(probability, iterations)
    skeleton_true = soft_skeleton(target, iterations)

    dims = tuple(range(1, target.dim()))
    precision = (torch.sum(skeleton_pred * target, dim=dims) + smooth) / (
        torch.sum(skeleton_pred, dim=dims) + smooth
    )
    sensitivity = (torch.sum(skeleton_true * probability, dim=dims) + smooth) / (
        torch.sum(skeleton_true, dim=dims) + smooth
    )
    cl_dice = 2.0 * precision * sensitivity / (precision + sensitivity + 1e-8)
    return (1.0 - cl_dice).mean()


def tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    smooth: float = 1.0,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Tversky loss, a Dice generalisation with asymmetric error weighting.

    ``alpha`` weights false positives and ``beta`` false negatives.  The default
    ``beta > alpha`` leans towards recall, which is what we want: a missed barb
    is a hard failure on the challenge's fine-structure criterion, whereas a
    couple of over-called pixels along an edge are comparatively cheap.
    """
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
    """Focal loss: down-weights the easy quiet-Sun pixels that dominate the disk."""
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
    """Relative weights of each term in :class:`FilamentLoss`."""

    bce: float = 1.0
    tversky: float = 1.0
    cl_dice: float = 1.0
    focal: float = 0.5
    spine: float = 0.5
    boundary: float = 0.5
    deep: float = 0.4


class FilamentLoss(nn.Module):
    """The complete multi-task objective used to train FilaNet.

    Args:
        weights: Relative weight of each term.
        tversky_alpha: False-positive weight in the Tversky term.
        tversky_beta: False-negative weight in the Tversky term.
        cl_dice_iterations: Thinning steps in the soft skeleton.
        pos_weight: Positive-class weight inside the cross-entropy term.
        cl_dice_warmup: Number of training steps over which the clDice weight is
            ramped in from zero.  Skeletonising a randomly initialised, nearly
            uniform prediction is meaningless and destabilises early training,
            so the topology term only switches on once the mask term has given
            it something with a shape.
    """

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
        """Compute the total loss and a breakdown for logging.

        Args:
            outputs: What :class:`~filaseg.models.filanet.FilaNet` returned.
            batch: The training batch, with ``mask``, ``spine``, ``boundary``,
                ``weight`` and ``valid`` entries.

        Returns:
            ``(total_loss, components)`` where ``components`` maps each term's
            name to its scalar value.
        """
        logits = outputs["mask"]
        assert isinstance(logits, torch.Tensor)
        target = batch["mask"]
        valid = batch.get("valid")
        weight = batch.get("weight")

        components: dict[str, float] = {}
        total = logits.new_zeros(())

        if self.weights.bce > 0:
            term = weighted_bce(logits, target, weight, valid, self.pos_weight)
            total = total + self.weights.bce * term
            components["bce"] = float(term.detach())

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
            term = cl_dice_loss(logits, target, self.cl_dice_iterations)
            total = total + self.weights.cl_dice * scale * term
            components["cl_dice"] = float(term.detach())

        if self.weights.spine > 0 and "spine" in outputs and "spine" in batch:
            spine_logits = outputs["spine"]
            assert isinstance(spine_logits, torch.Tensor)
            term = weighted_bce(spine_logits, batch["spine"], mask=valid)
            total = total + self.weights.spine * term
            components["spine"] = float(term.detach())

        if self.weights.boundary > 0 and "boundary" in outputs and "boundary" in batch:
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
                    F.adaptive_max_pool2d(valid, size) if valid is not None else None
                )
                deep_total = deep_total + weighted_bce(
                    deep_logits, small_target, mask=small_valid, pos_weight=self.pos_weight
                )
            deep_total = deep_total / len(deep)
            total = total + self.weights.deep * deep_total
            components["deep"] = float(deep_total.detach())

        if self.training:
            self._step += 1
        components["total"] = float(total.detach())
        return total, components
