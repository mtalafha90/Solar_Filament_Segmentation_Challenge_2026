"""Shared fixtures. Puts ``src`` on the path so tests run from a clean checkout."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filaseg.data.synthetic import generate_observation  # noqa: E402


@pytest.fixture(scope="session")
def observation():
    """One synthetic full-disk observation, shared across tests."""
    return generate_observation(size=256, n_filaments=5, n_sunspots=3, seed=42)


@pytest.fixture(scope="session")
def synthetic_dataset(tmp_path_factory):
    """A small MAGFiLO-style dataset written to a temporary directory."""
    import json

    sys.path.insert(0, str(ROOT / "scripts"))
    from make_synthetic_dataset import build

    out = tmp_path_factory.mktemp("dataset")
    coco = build(out, n_images=3, size=192, n_filaments=4, n_sunspots=2, seed=7)
    with (out / "annotations.json").open("w", encoding="utf-8") as handle:
        json.dump(coco, handle)
    return out
