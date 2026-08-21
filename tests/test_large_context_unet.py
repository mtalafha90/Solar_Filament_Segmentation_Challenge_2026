"""Regression tests for the native-1024 ResNet U-Net benchmark."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from conftest import require

torch = require("torch")
pytest.importorskip("torchvision")

from filaseg.losses import FilamentLoss, LossWeights, dice_loss  # noqa: E402
from filaseg.models.filanet import FilaNetConfig, build_model  # noqa: E402


def _mask(size: int = 32):
    target = torch.zeros(1, 1, size, size)
    target[:, :, 12:18, 4:28] = 1.0
    return target


def _logits(mask, confidence: float = 12.0):
    return mask * confidence - confidence / 2


def test_plain_unet_mode_has_no_attention_or_auxiliary_heads():
    model = build_model(
        FilaNetConfig(
            encoder="resnet18",
            pretrained=False,
            in_channels=2,
            bottleneck_attention=False,
            aux_heads=False,
            deep_supervision=False,
        )
    )
    assert model.attention is None
    assert model.edge_map is None

    outputs = model(torch.randn(1, 2, 128, 128))
    assert outputs["mask"].shape == (1, 1, 128, 128)
    assert set(outputs) == {"mask"}


def test_legacy_default_still_builds_attention():
    model = build_model(
        FilaNetConfig(base_width=8, depth=2, n_heads=2)
    )
    assert model.attention is not None
    assert model.edge_map is not None


def test_soft_dice_prefers_the_correct_mask():
    target = _mask()
    good = dice_loss(_logits(target), target)
    bad = dice_loss(_logits(1.0 - target), target)
    assert good.item() < 0.01
    assert good.item() < bad.item()


def test_distance_weighting_can_be_disabled_for_clean_bce_dice_control():
    target = _mask()
    logits = torch.zeros_like(target)
    valid = torch.ones_like(target)
    weights_a = torch.ones_like(target)
    weights_b = torch.full_like(target, 7.0)

    loss_weights = LossWeights(
        bce=0.5,
        dice=0.5,
        tversky=0.0,
        cl_dice=0.0,
        focal=0.0,
        spine=0.0,
        boundary=0.0,
        deep=0.0,
        use_distance_weight=False,
    )
    criterion = FilamentLoss(loss_weights, pos_weight=1.0, cl_dice_warmup=0)

    a, comp_a = criterion(
        {"mask": logits},
        {"mask": target, "valid": valid, "weight": weights_a},
    )
    b, comp_b = criterion(
        {"mask": logits},
        {"mask": target, "valid": valid, "weight": weights_b},
    )
    assert a.item() == pytest.approx(b.item())
    assert set(comp_a) == {"bce", "dice", "total"}
    assert set(comp_b) == {"bce", "dice", "total"}


def test_benchmark_config_locks_the_intended_ablation():
    cfg = yaml.safe_load(Path("configs/resnet34_unet_1024.yaml").read_text())
    assert cfg["patch_size"] == 1024
    assert cfg["model"]["encoder"] == "resnet34"
    assert cfg["model"]["pretrained"] is True
    assert cfg["model"]["bottleneck_attention"] is False
    assert cfg["model"]["aux_heads"] is False
    assert cfg["model"]["deep_supervision"] is False
    assert cfg["loss"]["bce"] == pytest.approx(0.5)
    assert cfg["loss"]["dice"] == pytest.approx(0.5)
    assert cfg["loss"]["use_distance_weight"] is False
