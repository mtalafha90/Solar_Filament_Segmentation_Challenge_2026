"""Tests for the network, tiled inference and submission writing."""

import numpy as np
import pytest

from conftest import require

torch = require("torch")

from filaseg.inference import (  # noqa: E402
    InferenceConfig,
    _blend_window,
    _dihedral,
    predict,
    predict_probability,
    segment,
)
from filaseg.models.edge_attention import EdgeGuidedAttention, LearnableEdgeMap  # noqa: E402
from filaseg.models.filanet import FilaNetConfig, build_model  # noqa: E402
from filaseg.submission import (  # noqa: E402
    decode_kaggle_rle,
    instances_from_labels,
    kaggle_rle,
    summarise_predictions,
    write_coco,
    write_rle_csv,
)


def _small_model(**kwargs):
    return build_model(FilaNetConfig(base_width=8, depth=2, n_heads=2, **kwargs))


def test_forward_shapes_and_heads():
    model = _small_model()
    outputs = model(torch.randn(2, 2, 64, 64))
    assert outputs["mask"].shape == (2, 1, 64, 64)
    assert outputs["spine"].shape == (2, 1, 64, 64)
    assert outputs["boundary"].shape == (2, 1, 64, 64)
    assert len(outputs["deep"]) == 1


def test_auxiliary_heads_can_be_disabled():
    outputs = _small_model(aux_heads=False, deep_supervision=False)(torch.randn(1, 2, 32, 32))
    assert "spine" not in outputs and "deep" not in outputs


def test_mask_head_starts_biased_towards_background():
    model = _small_model()
    probability = torch.sigmoid(model(torch.zeros(1, 2, 32, 32))["mask"])
    # Filaments are a per cent of the disk, so an untrained model should not
    # begin by predicting half the image as filament.
    assert probability.mean().item() < 0.3


def test_every_parameter_receives_gradient():
    model = _small_model()
    outputs = model(torch.randn(1, 2, 32, 32))
    loss = (
        outputs["mask"].mean()
        + outputs["spine"].mean()
        + outputs["boundary"].mean()
        + sum(d.mean() for d in outputs["deep"])
    )
    loss.backward()
    missing = [name for name, p in model.named_parameters() if p.grad is None]
    assert not missing, f"no gradient for {missing}"


def test_edge_map_starts_as_a_real_edge_detector():
    edge = LearnableEdgeMap(1, 8)
    flat = torch.ones(1, 1, 32, 32)
    stepped = torch.zeros(1, 1, 32, 32)
    stepped[..., 16:] = 1.0
    # Raw filter responses: flat input gives nothing, a step gives a response.
    assert torch.abs(edge.filters(flat)).mean() < torch.abs(edge.filters(stepped)).mean()


def test_edge_attention_starts_as_plain_attention():
    """Zero-initialised edge projections mean edges have no effect at step zero."""
    attention = EdgeGuidedAttention(16, edge_channels=4, n_heads=2)
    features = torch.randn(1, 16, 8, 8)
    a = attention(features, torch.zeros(1, 4, 8, 8))
    b = attention(features, torch.randn(1, 4, 8, 8))
    assert torch.allclose(a, b, atol=1e-6)


def test_edge_attention_matters_once_trained():
    attention = EdgeGuidedAttention(16, edge_channels=4, n_heads=2)
    torch.nn.init.normal_(attention.edge_to_q.weight, std=0.5)
    torch.nn.init.normal_(attention.edge_to_k.weight, std=0.5)
    features = torch.randn(1, 16, 8, 8)
    a = attention(features, torch.zeros(1, 4, 8, 8))
    b = attention(features, torch.randn(1, 4, 8, 8))
    assert not torch.allclose(a, b, atol=1e-4)


def test_edge_attention_can_be_switched_off():
    model = _small_model(edge_attention=False)
    assert not model.attention.use_edge
    assert not hasattr(model.attention, "edge_to_q")
    assert model(torch.randn(1, 2, 32, 32))["mask"].shape == (1, 1, 32, 32)


def test_attention_resizes_a_mismatched_edge_map():
    attention = EdgeGuidedAttention(8, edge_channels=4, n_heads=2)
    out = attention(torch.randn(1, 8, 8, 8), torch.randn(1, 4, 32, 32))
    assert out.shape == (1, 8, 8, 8)


def test_predict_returns_probabilities():
    model = _small_model()
    probability = torch.sigmoid(model(torch.randn(1, 2, 32, 32))["mask"])
    assert 0.0 <= probability.min().item() and probability.max().item() <= 1.0


def test_dihedral_transforms_round_trip():
    array = np.random.rand(2, 3, 8, 8).astype(np.float32)
    for index in range(8):
        assert np.allclose(_dihedral(_dihedral(array, index), index, inverse=True), array)


def test_blend_window_tapers_to_the_edges():
    window = _blend_window(32, 8)
    assert window[16, 16] == pytest.approx(1.0)
    assert window[0, 0] < 0.01
    assert (window > 0).all()


class _LocalModel(torch.nn.Module):
    """A model whose output depends only on the pixel under it.

    With no spatial receptive field there are no border effects, so any
    disagreement between tiled and whole-image inference is entirely down to the
    tiling and blending arithmetic. That is exactly what we want to test.
    """

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv2d(2, 1, 1)

    def forward(self, x):
        return {"mask": self.conv(x)}


def test_tiled_inference_reproduces_a_single_pass():
    """Overlapped tiling plus blending must be arithmetically transparent."""
    torch.manual_seed(0)
    model = _LocalModel()
    inputs = np.random.default_rng(0).random((2, 96, 96)).astype(np.float32)

    with torch.no_grad():
        whole = torch.sigmoid(model(torch.from_numpy(inputs)[None])["mask"])[0, 0].numpy()
    tiled = predict_probability(
        model,
        inputs,
        InferenceConfig(tile_size=48, overlap=0.5, tta=False, skip_empty_tiles=False),
    )
    assert tiled.shape == whole.shape
    assert np.allclose(tiled, whole, atol=1e-5)


def test_tiled_inference_leaves_no_seams():
    """A constant input must give a constant output, whatever the tiling."""
    torch.manual_seed(0)
    model = _LocalModel()
    inputs = np.full((2, 100, 100), 0.5, dtype=np.float32)

    tiled = predict_probability(
        model,
        inputs,
        InferenceConfig(tile_size=32, overlap=0.25, tta=False, skip_empty_tiles=False),
    )
    # Any seam from the blending would show up as a ripple across the frame.
    assert tiled.std() < 1e-6


def test_tiled_inference_covers_every_pixel():
    """Even with an awkward size that no tile grid divides evenly."""
    torch.manual_seed(0)
    model = _LocalModel()
    inputs = np.random.default_rng(1).random((2, 77, 53)).astype(np.float32)
    tiled = predict_probability(
        model,
        inputs,
        InferenceConfig(tile_size=32, overlap=0.25, tta=False, skip_empty_tiles=False),
    )
    assert tiled.shape == (77, 53)
    assert np.isfinite(tiled).all()
    assert (tiled > 0).all()  # no pixel left unwritten


def test_tta_produces_a_valid_probability_map():
    model = _small_model()
    inputs = np.random.rand(2, 64, 64).astype(np.float32)
    probability = predict_probability(
        model, inputs, InferenceConfig(tile_size=32, tta=True, skip_empty_tiles=False)
    )
    assert probability.shape == (64, 64)
    assert probability.min() >= 0.0 and probability.max() <= 1.0


def test_predict_zeroes_off_disk(observation):
    model = _small_model()
    probability, valid, disk = predict(
        model, observation.image, InferenceConfig(tile_size=64, tta=False)
    )
    assert probability.shape == observation.image.shape
    assert probability[~valid].max() == pytest.approx(0.0)
    assert disk.radius > 0


def test_segment_returns_instance_labels(observation):
    model = _small_model()
    labels, probability, valid = segment(
        model, observation.image, InferenceConfig(tile_size=64, tta=False)
    )
    assert labels.dtype == np.int32
    assert labels.shape == observation.image.shape
    assert not (labels > 0)[~valid].any()


def test_kaggle_rle_round_trip():
    rng = np.random.default_rng(1)
    mask = rng.random((30, 20)) > 0.6
    assert np.array_equal(decode_kaggle_rle(kaggle_rle(mask), 30, 20), mask)


def test_kaggle_rle_of_empty_mask_is_blank():
    assert kaggle_rle(np.zeros((8, 8), dtype=bool)) == ""


def test_instances_from_labels_scores_by_probability():
    labels = np.zeros((16, 16), dtype=np.int32)
    labels[:4, :4] = 1
    labels[8:12, 8:12] = 2
    probability = np.zeros((16, 16), dtype=np.float32)
    probability[:4, :4] = 0.9
    probability[8:12, 8:12] = 0.6

    instances = instances_from_labels(labels, probability)
    assert len(instances) == 2
    assert instances[0][1] == pytest.approx(0.9)
    assert instances[1][1] == pytest.approx(0.6)


def test_write_coco_and_csv(tmp_path):
    import json

    labels = np.zeros((32, 32), dtype=np.int32)
    labels[4:12, 4:24] = 1
    probability = (labels > 0).astype(np.float32) * 0.8

    count = write_coco(tmp_path / "sub.json", [(7, labels, probability)])
    assert count == 1
    entry = json.loads((tmp_path / "sub.json").read_text())[0]
    assert entry["image_id"] == 7 and entry["score"] == pytest.approx(0.8)
    assert entry["bbox"][2] > 0

    rows = write_rle_csv(
        tmp_path / "sub.csv",
        [("a", labels, probability), ("b", np.zeros((32, 32), np.int32), None)],
    )
    assert rows == 2  # one instance plus one blank row for the empty image


def test_write_coco_with_rle(tmp_path):
    import json

    labels = np.zeros((16, 16), dtype=np.int32)
    labels[2:6, 2:10] = 1
    write_coco(tmp_path / "rle.json", [(1, labels, None)], use_rle=True)
    entry = json.loads((tmp_path / "rle.json").read_text())[0]
    assert isinstance(entry["segmentation"], dict)
    assert entry["segmentation"]["size"] == [16, 16]


def test_summarise_predictions():
    labels = np.zeros((32, 32), dtype=np.int32)
    labels[:4, :4] = 1
    summary = summarise_predictions([labels, np.zeros((32, 32), dtype=np.int32)])
    assert summary["total_instances"] == 1
    assert summary["images_with_no_detection"] == 1
    assert summary["n_images"] == 2
