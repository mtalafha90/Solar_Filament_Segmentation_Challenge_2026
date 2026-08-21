"""End-to-end tests: a short training run and the full prediction pipeline."""

import numpy as np
import pytest

from conftest import require

torch = require("torch")

from filaseg.metrics import evaluate  # noqa: E402
from filaseg.models.filanet import FilaNetConfig  # noqa: E402
from filaseg.losses import LossWeights  # noqa: E402
from filaseg.train import (  # noqa: E402
    TrainConfig,
    build_scheduler,
    load_model,
    split_ids,
    train,
)


def test_split_is_disjoint_and_deterministic():
    train_a, val_a = split_ids(20, 0.2, seed=0)
    train_b, val_b = split_ids(20, 0.2, seed=0)
    assert train_a == train_b and val_a == val_b
    assert not set(train_a) & set(val_a)
    assert len(train_a) + len(val_a) == 20
    assert len(val_a) == 4


def test_grouped_split_never_leaks_a_physical_image():
    """All annotation records for one physical frame must stay on one side."""
    groups = ["a", "a", "b", "b", "b", "c", "d", "d", "e"]
    train_indices, val_indices = split_ids(
        len(groups), 0.4, seed=7, groups=groups
    )
    train_groups = {groups[i] for i in train_indices}
    val_groups = {groups[i] for i in val_indices}
    assert train_groups.isdisjoint(val_groups)
    assert sorted(train_indices + val_indices) == list(range(len(groups)))


def test_split_keeps_at_least_one_validation_image():
    _, val = split_ids(3, 0.01, seed=0)
    assert len(val) >= 1


def test_scheduler_warms_up_then_decays():
    model = torch.nn.Linear(2, 2)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = build_scheduler(optimiser, TrainConfig(epochs=10, warmup_epochs=2), 5)

    rates = []
    for _ in range(50):
        rates.append(optimiser.param_groups[0]["lr"])
        optimiser.step()
        scheduler.step()

    assert rates[0] < rates[9]  # warming up
    assert max(rates) == pytest.approx(1e-3, rel=0.01)
    assert rates[-1] < rates[len(rates) // 2]  # then decaying


@pytest.mark.slow
def test_training_reduces_loss_and_produces_a_usable_checkpoint(
    synthetic_dataset, tmp_path
):
    config = TrainConfig(
        annotations=str(synthetic_dataset / "annotations.json"),
        image_dir=str(synthetic_dataset / "images"),
        cache_dir=str(tmp_path / "cache"),
        output_dir=str(tmp_path / "run"),
        epochs=2,
        patch_size=64,
        batch_size=4,
        samples_per_epoch=16,
        val_fraction=0.34,
        device="cpu",
        amp=False,
        model=FilaNetConfig(base_width=8, depth=2, n_heads=2),
        thresholds=(0.4, 0.5),
        val_tile=128,   # must suit the model depth; check_geometry enforces it
    )
    best = train(config)

    assert "best_iou" in best
    checkpoint = tmp_path / "run" / "best.pt"
    assert checkpoint.exists()

    import json

    history = json.loads((tmp_path / "run" / "history.json").read_text())
    assert len(history) == 2
    assert history[-1]["train_total"] < history[0]["train_total"]

    model, threshold = load_model(checkpoint, "cpu")
    assert 0.0 < threshold <= 1.0
    with torch.no_grad():
        assert model(torch.randn(1, 2, 64, 64))["mask"].shape == (1, 1, 64, 64)


@pytest.mark.slow
def test_full_pipeline_from_raw_image_to_metrics(synthetic_dataset, tmp_path):
    """Raw image in, scored instance labels out, with no manual steps between."""
    from filaseg.data.dataset import MagfiloDataset
    from filaseg.inference import InferenceConfig, segment
    from filaseg.data.io import find_image, read_image
    from filaseg.models.filanet import build_model

    dataset = MagfiloDataset(
        synthetic_dataset / "annotations.json", synthetic_dataset / "images"
    )
    prepared = dataset[0]
    raw = read_image(find_image(synthetic_dataset / "images", prepared.file_name))

    model = build_model(FilaNetConfig(base_width=8, depth=2, n_heads=2))
    labels, probability, valid = segment(
        model, raw, InferenceConfig(tile_size=64, tta=False)
    )

    assert labels.shape == raw.shape
    assert probability.shape == raw.shape
    scores = evaluate(labels, prepared.instances, valid)
    # An untrained model will score badly; the point is that every metric is
    # produced and finite, so the pipeline is wired up correctly end to end.
    for key in ("iou", "cl_dice", "msiou", "hit_rate", "mean_pairwise_iou"):
        assert key in scores and np.isfinite(scores[key])


def test_classical_pipeline_needs_no_training(synthetic_dataset):
    """The classical detector must work straight from a raw image."""
    from filaseg.classical import detect
    from filaseg.data.dataset import MagfiloDataset
    from filaseg.data.io import find_image, read_image

    dataset = MagfiloDataset(
        synthetic_dataset / "annotations.json", synthetic_dataset / "images"
    )
    prepared = dataset[0]
    raw = read_image(find_image(synthetic_dataset / "images", prepared.file_name))

    labels = detect(raw)
    scores = evaluate(labels, prepared.instances, prepared.valid)
    assert labels.max() >= 1
    assert scores["iou"] > 0.2


def test_geometry_is_checked_before_training_starts():
    """A tile too large for the model's depth must fail now, not after an epoch."""
    from filaseg.train import check_geometry

    # The shipped combination is fine.
    check_geometry(TrainConfig(patch_size=512, val_tile=512,
                               model=FilaNetConfig(depth=4)))

    # A shallow model with a large validation tile is not, and validation only
    # runs once a whole epoch has been spent.
    with pytest.raises(ValueError, match="val_tile"):
        check_geometry(TrainConfig(patch_size=128, val_tile=512,
                                   model=FilaNetConfig(depth=2)))

    with pytest.raises(ValueError, match="patch_size"):
        check_geometry(TrainConfig(patch_size=1024, val_tile=128,
                                   model=FilaNetConfig(depth=2)))

    # And a size that does not divide by the downsampling factor.
    with pytest.raises(ValueError, match="divisible"):
        check_geometry(TrainConfig(patch_size=300, val_tile=128,
                                   model=FilaNetConfig(depth=4)))
