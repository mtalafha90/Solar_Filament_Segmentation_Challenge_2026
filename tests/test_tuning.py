"""Regression tests for reproducible post-processing tuning."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

SCRIPTS = Path("scripts").resolve()
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

spec = importlib.util.spec_from_file_location(
    "tune_postprocess", SCRIPTS / "tune_postprocess.py"
)
assert spec and spec.loader
tune_postprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tune_postprocess)


def test_evenly_spaced_subset_is_deterministic_and_not_tail_only():
    indices = list(range(10))
    selected = tune_postprocess._evenly_spaced(indices, 4)
    assert selected == [0, 2, 5, 7]
    assert selected != indices[-4:]
    assert tune_postprocess._evenly_spaced(indices, 0) == indices


def test_parameter_grid_skips_redundant_confidence_floors():
    args = SimpleNamespace(
        thresholds=[0.90, 0.95],
        min_confidence=[0.0, 0.92, 0.97],
        merge_gap=[40.0],
        min_area_fraction=[1.2e-4],
    )
    grid = tune_postprocess._parameter_grid(args)
    assert (0.90, 0.0, 40.0, 1.2e-4) in grid
    assert (0.90, 0.92, 40.0, 1.2e-4) in grid
    assert (0.95, 0.92, 40.0, 1.2e-4) not in grid
    assert (0.95, 0.97, 40.0, 1.2e-4) in grid


def test_probability_cache_namespace_changes_with_inference_inputs(tmp_path):
    checkpoint = tmp_path / "model.pt"
    checkpoint.write_bytes(b"checkpoint-a")

    class Record:
        def __init__(self, image_id: str):
            self.image_id = image_id

    dataset = SimpleNamespace(
        records=[Record("img-a"), Record("img-b")],
        group_keys=["frame-a", "frame-b"],
    )
    indices = [0, 1]

    base, _ = tune_postprocess.probability_cache_namespace(
        checkpoint, dataset, indices, 256, False
    )
    tile_changed, _ = tune_postprocess.probability_cache_namespace(
        checkpoint, dataset, indices, 512, False
    )
    tta_changed, _ = tune_postprocess.probability_cache_namespace(
        checkpoint, dataset, indices, 256, True
    )
    split_changed, _ = tune_postprocess.probability_cache_namespace(
        checkpoint, dataset, [0], 256, False
    )

    checkpoint.write_bytes(b"checkpoint-b")
    model_changed, _ = tune_postprocess.probability_cache_namespace(
        checkpoint, dataset, indices, 256, False
    )

    assert len({base, tile_changed, tta_changed, split_changed, model_changed}) == 5
