"""Regression tests for competition-alignment fixes."""

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from filaseg.losses import cl_dice_loss  # noqa: E402
from filaseg.models.edge_attention import _edge_kernels  # noqa: E402
from filaseg.models.filanet import FilaNetConfig, build_model  # noqa: E402
from filaseg.train import TrainConfig  # noqa: E402


def test_full_filanet_preserves_physical_edge_initialization():
    """The model-wide initializer must not overwrite the physical edge bank."""
    model = build_model(
        FilaNetConfig(in_channels=2, base_width=8, depth=2, edge_channels=8, n_heads=2)
    )
    expected = _edge_kernels().repeat(1, 2, 1, 1) / 2.0
    assert torch.allclose(model.edge_map.filters.weight.detach(), expected)


def test_full_filanet_edge_guidance_starts_at_zero():
    """FilaNet must really start as plain attention, as documented."""
    model = build_model(
        FilaNetConfig(in_channels=2, base_width=8, depth=2, edge_channels=8, n_heads=2)
    )
    assert torch.count_nonzero(model.attention.edge_to_q.weight).item() == 0
    assert torch.count_nonzero(model.attention.edge_to_k.weight).item() == 0


def test_cldice_ignores_predictions_outside_valid_disk():
    """Off-disk logits must not alter the topology loss."""
    target = torch.zeros(1, 1, 32, 32)
    target[..., 14:18, 8:24] = 1.0
    valid = torch.zeros_like(target)
    valid[..., 6:26, 6:26] = 1.0

    clean = target * 12.0 - 6.0
    noisy = clean.clone()
    noisy[valid == 0] = 12.0

    clean_loss = cl_dice_loss(clean, target, iterations=5, mask=valid)
    noisy_loss = cl_dice_loss(noisy, target, iterations=5, mask=valid)
    assert noisy_loss.item() == pytest.approx(clean_loss.item(), abs=1e-7)


def test_default_config_selects_matched_dice():
    """Both Python and YAML defaults must select by per-filament matched Dice."""
    assert TrainConfig().selection_metric == "matched_dice"
    text = Path("configs/default.yaml").read_text(encoding="utf-8")
    assert "selection_metric: matched_dice" in text
