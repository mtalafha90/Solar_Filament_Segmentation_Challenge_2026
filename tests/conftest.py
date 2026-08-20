"""Shared fixtures. Puts ``src`` on the path so tests run from a clean checkout."""

import importlib
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from filaseg.data.synthetic import generate_observation  # noqa: E402


def require(module: str):
    """Import a module, or skip the test if it cannot be used.

    ``pytest.importorskip`` only skips when a module is *absent*. A module that
    is installed but broken -- a PyTorch whose CUDA library fails to resolve a
    symbol, say, which happens easily when conda and pip libraries are mixed --
    raises ``ImportError`` instead, and at module scope that aborts collection
    for the whole file. Everything that does not need the module then goes
    unrun, which is exactly when you most want the rest of the suite to report.

    Args:
        module: Name of the module to import.

    Returns:
        The imported module.
    """
    try:
        return importlib.import_module(module)
    except Exception as error:  # noqa: BLE001 - any failure means unusable
        pytest.skip(f"{module} is unavailable: {error}", allow_module_level=True)


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
