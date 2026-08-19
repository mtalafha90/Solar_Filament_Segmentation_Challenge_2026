"""Tests for the training objective."""

import pytest

torch = pytest.importorskip("torch")

from filaseg.losses import (  # noqa: E402
    FilamentLoss,
    LossWeights,
    cl_dice_loss,
    focal_loss,
    soft_dilate,
    soft_erode,
    soft_skeleton,
    tversky_loss,
    weighted_bce,
)


def _bar(width=8, length=48, size=64):
    mask = torch.zeros(1, 1, size, size)
    top = (size - width) // 2
    mask[0, 0, top : top + width, 8 : 8 + length] = 1.0
    return mask


def _logits(mask, confidence=12.0):
    return mask * confidence - confidence / 2


def test_soft_erode_and_dilate_move_in_opposite_directions():
    mask = _bar()
    assert soft_erode(mask).sum() < mask.sum()
    assert soft_dilate(mask).sum() > mask.sum()


def test_soft_skeleton_thins_a_bar():
    mask = _bar(width=8)
    skeleton = soft_skeleton(mask, iterations=10)
    assert skeleton.sum() < mask.sum() / 2
    # The remaining mass must lie along the bar's centre line.
    rows = (skeleton[0, 0] > 0.5).sum(dim=1)
    occupied = torch.nonzero(rows).flatten()
    assert len(occupied) <= 4
    assert 26 <= occupied.float().mean().item() <= 38


def test_soft_skeleton_of_empty_is_empty():
    assert soft_skeleton(torch.zeros(1, 1, 32, 32), 5).sum() == pytest.approx(0.0)


def test_cl_dice_is_zero_for_a_perfect_prediction():
    mask = _bar()
    assert cl_dice_loss(_logits(mask), mask).item() < 0.01


def test_cl_dice_punishes_a_deleted_barb_harder_than_tversky():
    truth = _bar(width=6, length=48)
    truth[0, 0, 20:29, 30:33] = 1.0  # a barb
    without = truth.clone()
    without[0, 0, 20:29, 30:33] = 0.0

    cl_full = cl_dice_loss(_logits(truth), truth).item()
    cl_cut = cl_dice_loss(_logits(without), truth).item()
    tv_full = tversky_loss(_logits(truth), truth).item()
    tv_cut = tversky_loss(_logits(without), truth).item()

    # Both must notice; clDice must react proportionally more strongly.
    assert cl_cut > cl_full and tv_cut > tv_full
    assert (cl_cut / max(cl_full, 1e-6)) > (tv_cut / max(tv_full, 1e-6))


def test_tversky_beta_controls_the_recall_bias():
    truth = _bar()
    under = truth.clone()
    under[0, 0, :, 30:] = 0.0  # a prediction that misses half the filament

    recall_biased = tversky_loss(_logits(under), truth, alpha=0.3, beta=0.7).item()
    precision_biased = tversky_loss(_logits(under), truth, alpha=0.7, beta=0.3).item()
    # Weighting false negatives more heavily must punish under-segmentation more.
    assert recall_biased > precision_biased


def test_weighted_bce_respects_weights_and_masks():
    logits = torch.zeros(1, 1, 8, 8)
    target = torch.ones(1, 1, 8, 8)
    plain = weighted_bce(logits, target).item()
    weights = torch.full((1, 1, 8, 8), 2.0)
    assert weighted_bce(logits, target, weight=weights).item() == pytest.approx(2 * plain)

    valid = torch.zeros(1, 1, 8, 8)
    valid[0, 0, :4, :4] = 1.0
    assert weighted_bce(logits, target, mask=valid).item() == pytest.approx(plain, rel=1e-5)


def test_focal_loss_down_weights_easy_pixels():
    target = torch.zeros(1, 1, 8, 8)
    easy = torch.full((1, 1, 8, 8), -8.0)  # confidently, correctly negative
    hard = torch.zeros(1, 1, 8, 8)  # completely unsure
    assert focal_loss(easy, target).item() < focal_loss(hard, target).item()


def test_filament_loss_returns_a_breakdown():
    mask = _bar()
    batch = {
        "mask": mask,
        "spine": mask * 0.5,
        "boundary": mask * 0.2,
        "weight": torch.ones_like(mask),
        "valid": torch.ones_like(mask),
    }
    outputs = {
        "mask": _logits(mask),
        "spine": _logits(mask * 0.5),
        "boundary": _logits(mask * 0.2),
        "deep": [torch.zeros(1, 1, 16, 16)],
    }
    criterion = FilamentLoss(cl_dice_warmup=0)
    total, components = criterion(outputs, batch)

    assert torch.isfinite(total)
    for key in ("bce", "tversky", "focal", "cl_dice", "spine", "boundary", "deep", "total"):
        assert key in components


def test_filament_loss_is_lower_for_a_better_prediction():
    mask = _bar()
    batch = {"mask": mask, "weight": torch.ones_like(mask), "valid": torch.ones_like(mask)}
    criterion = FilamentLoss(cl_dice_warmup=0)
    good, _ = criterion({"mask": _logits(mask)}, batch)
    bad, _ = criterion({"mask": _logits(1.0 - mask)}, batch)
    assert good.item() < bad.item()


def test_zero_weights_switch_terms_off():
    mask = _bar()
    batch = {"mask": mask, "weight": torch.ones_like(mask), "valid": torch.ones_like(mask)}
    weights = LossWeights(bce=1.0, tversky=0.0, cl_dice=0.0, focal=0.0,
                          spine=0.0, boundary=0.0, deep=0.0)
    _, components = FilamentLoss(weights, cl_dice_warmup=0)({"mask": _logits(mask)}, batch)
    assert "tversky" not in components and "cl_dice" not in components


def test_cl_dice_warmup_ramps_in():
    mask = _bar()
    batch = {"mask": mask, "weight": torch.ones_like(mask), "valid": torch.ones_like(mask)}
    criterion = FilamentLoss(cl_dice_warmup=10)
    criterion.train()
    assert criterion._cl_dice_scale() == pytest.approx(0.0)
    for _ in range(10):
        criterion({"mask": _logits(1.0 - mask)}, batch)
    assert criterion._cl_dice_scale() == pytest.approx(1.0)


def test_loss_is_differentiable():
    mask = _bar()
    logits = (_logits(mask)).clone().requires_grad_(True)
    batch = {"mask": mask, "weight": torch.ones_like(mask), "valid": torch.ones_like(mask)}
    total, _ = FilamentLoss(cl_dice_warmup=0)({"mask": logits}, batch)
    total.backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
