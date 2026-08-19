"""PyTorch datasets for filament segmentation.

Two things drive the design here.

**Severe class imbalance.**  Filaments cover well under one per cent of the
solar disk.  Sampling patches uniformly would spend almost all of the training
budget on empty quiet Sun, so :class:`FilamentPatchDataset` samples most patches
centred on an annotated filament and the rest at random.  The random share is
not optional: it is what teaches the model to reject sunspots and plage.

**Full-disk context at full resolution.**  Barbs are only a few pixels wide, so
downsampling a 2048-pixel frame to fit a network destroys exactly the structures
being scored.  We therefore train on full-resolution crops and stitch them back
together at inference.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

try:  # torch is only needed for the learned model, not the classical baseline
    import torch
    from torch.utils.data import Dataset

    _TORCH = True
except ImportError:  # pragma: no cover - exercised only without torch
    _TORCH = False

    class Dataset:  # type: ignore[no-redef]
        """Minimal stand-in so this module imports without torch installed."""


from ..preprocessing.disk import SolarDisk
from ..preprocessing.photometry import preprocess
from .coco import ImageId, ImageRecord, load_coco, normalise_id, rescale_record
from .io import find_image, read_image
from .targets import boundary_map, distance_weight, spine_heatmap


@dataclass
class PreparedObservation:
    """One observation, preprocessed and with all supervision targets attached."""

    image: np.ndarray  # (H, W) float32, filaments bright
    valid: np.ndarray  # (H, W) bool, on-disk mask
    mu: np.ndarray  # (H, W) float32, cosine of heliocentric angle
    mask: np.ndarray  # (H, W) bool, filament ground truth
    instances: np.ndarray  # (H, W) int32 label map
    spine: np.ndarray  # (H, W) float32 centreline heatmap
    boundary: np.ndarray  # (H, W) float32
    weight: np.ndarray  # (H, W) float32 per-pixel loss weights
    disk: SolarDisk
    image_id: ImageId = 0
    file_name: str = ""

    def input_stack(self) -> np.ndarray:
        """Model input: the flattened image plus a geometry channel.

        The ``mu`` channel tells the network how far round the limb a pixel
        sits.  That matters because filaments are foreshortened near the limb
        and because it implicitly marks the off-disk region as out of bounds.
        """
        return np.stack([self.image, self.mu], axis=0).astype(np.float32)


def prepare_observation(
    image: np.ndarray,
    record: ImageRecord | None = None,
    mask: np.ndarray | None = None,
    instances: np.ndarray | None = None,
    spine_points: Sequence[np.ndarray | None] | None = None,
    disk_fraction: float = 0.995,
) -> PreparedObservation:
    """Preprocess an image and build every supervision target from it.

    Args:
        image: Raw full-disk image.
        record: COCO record supplying masks and spines. Optional.
        mask: Explicit semantic mask, used when ``record`` is absent.
        instances: Explicit instance label map.
        spine_points: Per-instance spine polylines in ``(y, x)`` order.
        disk_fraction: Fraction of the solar radius treated as valid.

    Returns:
        A fully populated :class:`PreparedObservation`.
    """
    processed, valid, disk = preprocess(image, disk_fraction=disk_fraction)
    radius_map = disk.radial_map(processed.shape)
    mu = np.sqrt(np.clip(1.0 - np.clip(radius_map, 0.0, 1.0) ** 2, 0.0, 1.0))
    mu = (mu * valid).astype(np.float32)

    height, width = processed.shape
    if record is not None:
        instance_map = record.instance_map()
        semantic = instance_map > 0
        spines = [annotation.spine for annotation in record.annotations]
    else:
        semantic = (
            np.zeros((height, width), dtype=bool) if mask is None else mask.astype(bool)
        )
        instance_map = (
            semantic.astype(np.int32) if instances is None else instances.astype(np.int32)
        )
        spines = list(spine_points) if spine_points is not None else []

    # Annotations must not spill off the disk.
    semantic = semantic & valid
    instance_map = instance_map * valid

    # Build the centreline heatmap instance by instance so that neighbouring
    # filaments do not get a single merged skeleton.
    spine = np.zeros((height, width), dtype=np.float32)
    labels = np.unique(instance_map)
    for label in labels[labels > 0]:
        piece = instance_map == label
        points = spines[label - 1] if 0 <= label - 1 < len(spines) else None
        np.maximum(spine, spine_heatmap(piece, points), out=spine)

    return PreparedObservation(
        image=processed,
        valid=valid,
        mu=mu,
        mask=semantic,
        instances=instance_map.astype(np.int32),
        spine=spine,
        boundary=boundary_map(semantic),
        weight=distance_weight(semantic, valid),
        disk=disk,
        image_id=record.image_id if record is not None else 0,
        file_name=record.file_name if record is not None else "",
    )


class MagfiloDataset:
    """Lazily loads and preprocesses a MAGFiLO-style dataset, with a disk cache.

    Preprocessing a 2048-pixel frame costs a second or two, which is far too
    slow to repeat every epoch.  The first pass writes each prepared observation
    to ``cache_dir`` as a compressed ``.npz``; later passes memory-map it back.
    """

    def __init__(
        self,
        annotations: str | Path,
        image_dir: str | Path,
        cache_dir: str | Path | None = None,
        image_ids: Sequence[ImageId] | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._warned_about_size = False
        records, self.meta = load_coco(annotations)
        if image_ids is not None:
            wanted = {normalise_id(i) for i in image_ids}
            records = [r for r in records if r.image_id in wanted]
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    @property
    def image_ids(self) -> list[ImageId]:
        return [record.image_id for record in self.records]

    def _cache_path(self, record: ImageRecord) -> Path | None:
        """Cache file for one observation.

        Image ids are not necessarily integers -- MAGFiLO uses the original GONG
        frame name -- so the id is sanitised into something safe for a filename
        and given a short hash suffix, which keeps two ids that sanitise to the
        same text from colliding.
        """
        if self.cache_dir is None:
            return None
        raw = str(record.image_id)
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in raw)[:80]
        digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:8]
        return self.cache_dir / f"{safe}_{digest}.npz"

    def __getitem__(self, index: int) -> PreparedObservation:
        record = self.records[index]
        cache_path = self._cache_path(record)

        if cache_path is not None and cache_path.exists():
            with np.load(cache_path) as blob:
                return PreparedObservation(
                    image=blob["image"],
                    valid=blob["valid"].astype(bool),
                    mu=blob["mu"],
                    mask=blob["mask"].astype(bool),
                    instances=blob["instances"],
                    spine=blob["spine"],
                    boundary=blob["boundary"],
                    weight=blob["weight"],
                    disk=SolarDisk(*[float(v) for v in blob["disk"]]),
                    image_id=normalise_id(blob["image_id"].item()),
                    file_name=str(blob["file_name"]),
                )

        image = read_image(find_image(self.image_dir, record.file_name))
        if record.height == 0 or record.width == 0:
            record.height, record.width = image.shape
        elif (record.height, record.width) != image.shape:
            # The distributed images are not the size the annotations were drawn
            # on. Scale the annotations rather than the image, so we keep full
            # resolution: barbs do not survive downsampling.
            if not self._warned_about_size:
                import warnings

                warnings.warn(
                    f"annotation size {(record.height, record.width)} does not match "
                    f"image size {image.shape}; rescaling annotations to fit",
                    stacklevel=2,
                )
                self._warned_about_size = True
            rescale_record(record, image.shape[0], image.shape[1])
        prepared = prepare_observation(image, record=record)

        if cache_path is not None:
            np.savez_compressed(
                cache_path,
                image=prepared.image,
                valid=prepared.valid,
                mu=prepared.mu,
                mask=prepared.mask,
                instances=prepared.instances,
                spine=prepared.spine,
                boundary=prepared.boundary,
                weight=prepared.weight,
                disk=np.array(
                    [prepared.disk.centre_y, prepared.disk.centre_x, prepared.disk.radius],
                    dtype=np.float64,
                ),
                image_id=str(prepared.image_id),
                file_name=prepared.file_name,
            )
        return prepared


class FilamentPatchDataset(Dataset):
    """Random crops from prepared observations, biased towards filaments.

    Args:
        source: The :class:`MagfiloDataset` to draw from.
        patch_size: Side length of the square crops.
        samples_per_epoch: How many crops make up one epoch.
        positive_fraction: Share of crops centred on an annotated filament.
            The remainder are drawn uniformly from the disk, which is what
            teaches the model to reject sunspots and other dark distractors.
        augment: Apply dihedral and photometric augmentation.
        seed: Seed for reproducible sampling.
    """

    def __init__(
        self,
        source: MagfiloDataset,
        patch_size: int = 256,
        samples_per_epoch: int = 2000,
        positive_fraction: float = 0.7,
        augment: bool = True,
        seed: int = 0,
    ) -> None:
        if not _TORCH:  # pragma: no cover
            raise ImportError("FilamentPatchDataset needs PyTorch installed")
        self.source = source
        self.patch_size = int(patch_size)
        self.samples_per_epoch = int(samples_per_epoch)
        self.positive_fraction = float(positive_fraction)
        self.augment = bool(augment)
        self.seed = int(seed)
        self._epoch = 0
        self._cache: dict[int, PreparedObservation] = {}
        self._positions: dict[int, np.ndarray] = {}

    def set_epoch(self, epoch: int) -> None:
        """Change the sampling seed between epochs so crops differ each pass."""
        self._epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def _observation(self, index: int) -> PreparedObservation:
        if index not in self._cache:
            self._cache[index] = self.source[index]
        return self._cache[index]

    def _filament_positions(self, index: int, prepared: PreparedObservation) -> np.ndarray:
        if index not in self._positions:
            positions = np.argwhere(prepared.mask)
            if len(positions) > 20000:
                # Thinning keeps the memory bounded without biasing the sample.
                step = len(positions) // 20000 + 1
                positions = positions[::step]
            self._positions[index] = positions
        return self._positions[index]

    def __getitem__(self, item: int) -> dict[str, "torch.Tensor"]:
        rng = np.random.default_rng(
            (self.seed * 1_000_003 + self._epoch * 10_007 + item) % (2**32)
        )
        index = int(rng.integers(0, len(self.source)))
        prepared = self._observation(index)
        height, width = prepared.image.shape
        half = self.patch_size // 2

        positions = self._filament_positions(index, prepared)
        want_positive = rng.random() < self.positive_fraction and len(positions) > 0
        if want_positive:
            centre = positions[int(rng.integers(0, len(positions)))]
            # Jitter so filaments are not always dead centre in the crop.
            jitter = rng.integers(-half // 2, half // 2 + 1, size=2)
            centre_y = int(centre[0] + jitter[0])
            centre_x = int(centre[1] + jitter[1])
        else:
            disk = prepared.disk
            angle = rng.uniform(0.0, 2.0 * np.pi)
            radius = disk.radius * np.sqrt(rng.uniform(0.0, 1.0))
            centre_y = int(disk.centre_y + radius * np.sin(angle))
            centre_x = int(disk.centre_x + radius * np.cos(angle))

        top = int(np.clip(centre_y - half, 0, max(0, height - self.patch_size)))
        left = int(np.clip(centre_x - half, 0, max(0, width - self.patch_size)))
        window = (slice(top, top + self.patch_size), slice(left, left + self.patch_size))

        fields = {
            "image": prepared.image[window],
            "mu": prepared.mu[window],
            "mask": prepared.mask[window].astype(np.float32),
            "spine": prepared.spine[window],
            "boundary": prepared.boundary[window],
            "weight": prepared.weight[window],
            "valid": prepared.valid[window].astype(np.float32),
        }
        fields = {key: _pad_to(value, self.patch_size) for key, value in fields.items()}

        if self.augment:
            fields = _augment(fields, rng)

        inputs = np.stack([fields["image"], fields["mu"]], axis=0)
        return {
            "input": torch.from_numpy(np.ascontiguousarray(inputs, dtype=np.float32)),
            "mask": torch.from_numpy(
                np.ascontiguousarray(fields["mask"], dtype=np.float32)
            )[None],
            "spine": torch.from_numpy(
                np.ascontiguousarray(fields["spine"], dtype=np.float32)
            )[None],
            "boundary": torch.from_numpy(
                np.ascontiguousarray(fields["boundary"], dtype=np.float32)
            )[None],
            "weight": torch.from_numpy(
                np.ascontiguousarray(fields["weight"], dtype=np.float32)
            )[None],
            "valid": torch.from_numpy(
                np.ascontiguousarray(fields["valid"], dtype=np.float32)
            )[None],
        }


def _pad_to(array: np.ndarray, size: int) -> np.ndarray:
    """Zero-pad a crop up to ``size`` if it fell off the edge of the frame."""
    if array.shape == (size, size):
        return array
    out = np.zeros((size, size), dtype=array.dtype)
    out[: array.shape[0], : array.shape[1]] = array
    return out


def _augment(fields: dict[str, np.ndarray], rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Dihedral and photometric augmentation.

    Flips and quarter turns are safe here: they are exact symmetries of the
    imaging geometry once limb darkening has been divided out, and there is
    nothing in a local crop that fixes an absolute orientation.  The intensity
    jitter stands in for the seeing and transparency differences between the
    six GONG sites.
    """
    if rng.random() < 0.5:
        fields = {key: np.fliplr(value) for key, value in fields.items()}
    if rng.random() < 0.5:
        fields = {key: np.flipud(value) for key, value in fields.items()}
    turns = int(rng.integers(0, 4))
    if turns:
        fields = {key: np.rot90(value, turns) for key, value in fields.items()}

    # Photometric jitter applies to the image channel only; the geometry
    # channel and every target must stay untouched.
    image = fields["image"].astype(np.float32).copy()
    image = image * rng.uniform(0.85, 1.15) + rng.uniform(-0.06, 0.06)
    if rng.random() < 0.3:
        image = image + rng.normal(0.0, 0.02, size=image.shape).astype(np.float32)
    fields["image"] = np.clip(image, 0.0, 1.5)
    return fields
