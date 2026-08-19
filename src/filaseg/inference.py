"""Running a trained model over a whole solar disk.

A 2048-pixel GONG frame will not fit through the network in one piece at a
sensible batch size, and downsampling it is not an option: barbs are a few
pixels wide, so any reduction destroys precisely what is being scored.  The
frame is therefore processed in overlapping tiles and blended back together.

Two details make the seams invisible:

* tiles overlap by a good fraction of their width, so every pixel is predicted
  several times with different amounts of surrounding context, and
* the blend is weighted by a raised-cosine window that falls to nearly zero at
  each tile's edge, so a pixel's prediction is dominated by the tile that saw
  the most context around it.

Test-time augmentation over the eight dihedral symmetries is applied by
default.  It costs eight forward passes and reliably recovers a little of the
fine structure, because a barb that one orientation misses another usually
catches.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing.disk import SolarDisk
from .preprocessing.photometry import preprocess


@dataclass
class InferenceConfig:
    """Settings for :func:`predict`."""

    tile_size: int = 512
    """Side length of each tile. Must be divisible by 2**depth of the network."""
    overlap: float = 0.25
    """Tile overlap as a fraction of the tile size."""
    batch_size: int = 4
    tta: bool = True
    """Average over the eight dihedral symmetries."""
    device: str = "cpu"
    skip_empty_tiles: bool = True
    """Skip tiles that contain no on-disk pixels, which is most of the corners."""


def _blend_window(size: int, edge: int) -> np.ndarray:
    """A separable raised-cosine window that tapers to nearly zero at the edges."""
    ramp = np.ones(size, dtype=np.float32)
    if edge > 0:
        taper = 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, edge + 2)[1:-1]))
        ramp[:edge] = taper
        ramp[-edge:] = taper[::-1]
    window = np.outer(ramp, ramp)
    # A small floor stops any pixel getting zero total weight.
    return np.maximum(window, 1e-3).astype(np.float32)


def _dihedral(array: np.ndarray, index: int, inverse: bool = False) -> np.ndarray:
    """Apply (or undo) one of the eight dihedral symmetries to a ``(..., H, W)`` array."""
    turns = index % 4
    flip = index >= 4
    if not inverse:
        if flip:
            array = np.flip(array, axis=-1)
        return np.rot90(array, turns, axes=(-2, -1))
    array = np.rot90(array, -turns, axes=(-2, -1))
    if flip:
        array = np.flip(array, axis=-1)
    return array


def predict_probability(
    model,
    inputs: np.ndarray,
    config: InferenceConfig | None = None,
) -> np.ndarray:
    """Run a model over a large multi-channel input, tile by tile.

    Args:
        model: A :class:`~filaseg.models.filanet.FilaNet` (or anything whose
            call returns a dict with a ``mask`` entry of logits).
        inputs: Array of shape ``(C, H, W)``.
        config: Tiling and augmentation settings.

    Returns:
        A ``(H, W)`` float32 array of filament probabilities.
    """
    import torch

    config = config or InferenceConfig()
    device = torch.device(config.device)
    model = model.to(device)
    model.eval()

    channels, height, width = inputs.shape
    tile = int(config.tile_size)
    step = max(1, int(round(tile * (1.0 - config.overlap))))
    edge = max(1, int(round(tile * config.overlap / 2)))
    window = _blend_window(tile, edge)

    # Pad so that every pixel is covered by at least one whole tile.
    pad_y = max(0, tile - height) + (-(height - tile) % step if height > tile else 0)
    pad_x = max(0, tile - width) + (-(width - tile) % step if width > tile else 0)
    padded = np.pad(inputs, ((0, 0), (0, pad_y), (0, pad_x)), mode="reflect")
    _, padded_h, padded_w = padded.shape

    accumulator = np.zeros((padded_h, padded_w), dtype=np.float32)
    weights = np.zeros((padded_h, padded_w), dtype=np.float32)

    tops = list(range(0, padded_h - tile + 1, step))
    lefts = list(range(0, padded_w - tile + 1, step))
    positions = [(t, l) for t in tops for l in lefts]

    if config.skip_empty_tiles:
        # The geometry channel is zero off-disk, so an all-zero tile is sky.
        positions = [
            (t, l)
            for t, l in positions
            if np.any(padded[:, t : t + tile, l : l + tile] != 0)
        ]

    n_transforms = 8 if config.tta else 1

    with torch.no_grad():
        for start in range(0, len(positions), config.batch_size):
            chunk = positions[start : start + config.batch_size]
            patches = np.stack(
                [padded[:, t : t + tile, l : l + tile] for t, l in chunk]
            ).astype(np.float32)

            summed = np.zeros((len(chunk), tile, tile), dtype=np.float32)
            for transform in range(n_transforms):
                augmented = (
                    _dihedral(patches, transform) if n_transforms > 1 else patches
                )
                batch = torch.from_numpy(np.ascontiguousarray(augmented)).to(device)
                logits = model(batch)["mask"]
                probability = torch.sigmoid(logits)[:, 0].cpu().numpy()
                if n_transforms > 1:
                    probability = _dihedral(probability, transform, inverse=True)
                summed += np.ascontiguousarray(probability)
            summed /= n_transforms

            for (top, left), prediction in zip(chunk, summed):
                accumulator[top : top + tile, left : left + tile] += prediction * window
                weights[top : top + tile, left : left + tile] += window

    probability = accumulator / np.maximum(weights, 1e-6)
    return probability[:height, :width].astype(np.float32)


def predict(
    model,
    image: np.ndarray,
    config: InferenceConfig | None = None,
    disk: SolarDisk | None = None,
) -> tuple[np.ndarray, np.ndarray, SolarDisk]:
    """Preprocess a raw full-disk image and predict filament probabilities.

    Args:
        model: The trained network.
        image: Raw full-disk image.
        config: Tiling and augmentation settings.
        disk: Pre-computed disk geometry; detected automatically when omitted.

    Returns:
        ``(probability, valid, disk)``.  Probabilities off the disk are set to
        zero, because a filament cannot be observed there.
    """
    processed, valid, fitted = preprocess(image, disk=disk)
    radius_map = fitted.radial_map(processed.shape)
    mu = np.sqrt(np.clip(1.0 - np.clip(radius_map, 0.0, 1.0) ** 2, 0.0, 1.0))
    inputs = np.stack([processed, (mu * valid).astype(np.float32)], axis=0)

    probability = predict_probability(model, inputs, config)
    return (probability * valid).astype(np.float32), valid, fitted


def segment(
    model,
    image: np.ndarray,
    config: InferenceConfig | None = None,
    instance_config=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full prediction pipeline: preprocess, run the model, extract instances.

    Returns:
        ``(labels, probability, valid)``.
    """
    from .postprocess.instances import extract_instances

    probability, valid, _ = predict(model, image, config)
    labels = extract_instances(probability, valid, instance_config)
    return labels, probability, valid
