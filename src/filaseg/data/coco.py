"""Reading MAGFiLO-style COCO annotations.

MAGFiLO ships one JSON file in the COCO layout, with the usual top-level keys
``info``, ``licenses``, ``categories``, ``images`` and ``annotations``.  Each
filament annotation carries a polygon segmentation, a bounding box, a spine
(the curve running along the filament's length) and a magnetic chirality label.

Different releases spell the extra fields slightly differently, so the reader
below looks for a few plausible key names rather than insisting on one.  It
also decodes RLE segmentations, which some COCO exporters emit instead of
polygons, so the loader keeps working if the organisers change the export.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

# Keys that have been seen to hold the spine and chirality in COCO exports of
# this dataset.  Order matters only in that the first match wins.
SPINE_KEYS = ("spine", "spines", "spine_points", "skeleton", "keypoints")
CHIRALITY_KEYS = ("chirality", "chirality_label", "magnetic_chirality", "handedness")

CHIRALITY_NAMES = {0: "unknown", 1: "sinistral", 2: "dextral"}

# MAGFiLO does not store chirality in a field of its own: it encodes it in the
# COCO category, with names 'Left', 'Right', 'Unidentifiable' and 'Ambiguous'.
# Reading only a 'chirality' key therefore reports every filament as unknown and
# quietly throws away a label that took a thousand person-hours to produce.
CHIRALITY_FROM_CATEGORY = {
    "left": 1,
    "sinistral": 1,
    "right": 2,
    "dextral": 2,
    "unidentifiable": 0,
    "ambiguous": 0,
    "unknown": 0,
    "filament": 0,
}

# COCO nominally uses integer ids, but real releases often do not. MAGFiLO keys
# its observations by the original GONG frame name, for example
# '040301-20140609195854Bh', which carries the site and timestamp. Those ids are
# meaningful and must survive round-tripping into a submission, so we keep them
# exactly as given rather than forcing them to integers.
ImageId = int | str


def normalise_id(value: Any) -> ImageId:
    """Keep an id as an int when it truly is one, otherwise as a string.

    Integer-valued ids stay integers so that plain COCO files behave exactly as
    before; anything else is preserved verbatim.
    """
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float) and float(value).is_integer():
        return int(value)
    text = str(value)
    # A string of digits is an integer id written as text.
    if text.lstrip("-").isdigit():
        try:
            return int(text)
        except ValueError:
            return text
    return text


@dataclass
class FilamentAnnotation:
    """One annotated filament."""

    annotation_id: ImageId
    image_id: ImageId
    bbox: tuple[float, float, float, float]  # COCO order: x, y, width, height
    segmentation: Any
    area: float = 0.0
    spine: np.ndarray | None = None  # (N, 2) array of (y, x) points
    chirality: int = 0
    category_id: ImageId = 1

    def mask(self, height: int, width: int) -> np.ndarray:
        """Rasterise this annotation into a boolean mask."""
        return decode_segmentation(self.segmentation, height, width)


@dataclass
class ImageRecord:
    """One annotated observation."""

    image_id: ImageId
    file_name: str
    height: int
    width: int
    annotations: list[FilamentAnnotation] = field(default_factory=list)
    extra: dict[str, Any] = field(default_factory=dict)

    def semantic_mask(self) -> np.ndarray:
        """Union of every filament in this observation."""
        out = np.zeros((self.height, self.width), dtype=bool)
        for annotation in self.annotations:
            out |= annotation.mask(self.height, self.width)
        return out

    def instance_map(self) -> np.ndarray:
        """Integer label map, 0 for background and 1..N for filaments.

        Later annotations overwrite earlier ones where they overlap, which is
        the standard COCO convention and is harmless here because filaments
        rarely overlap.
        """
        out = np.zeros((self.height, self.width), dtype=np.int32)
        for index, annotation in enumerate(self.annotations, start=1):
            out[annotation.mask(self.height, self.width)] = index
        return out

    def instance_masks(self) -> list[np.ndarray]:
        """One boolean mask per filament, in annotation order."""
        return [a.mask(self.height, self.width) for a in self.annotations]


def polygons_to_mask(
    polygons: Sequence[Sequence[float]], height: int, width: int
) -> np.ndarray:
    """Rasterise COCO polygons (flat ``[x0, y0, x1, y1, ...]`` lists) to a mask.

    Multiple polygons in one annotation are unioned, which is the COCO
    convention for a single object made of several disconnected parts.
    """
    from skimage.draw import polygon as draw_polygon

    mask = np.zeros((height, width), dtype=bool)
    for ring in polygons:
        coords = np.asarray(ring, dtype=np.float64).ravel()
        if coords.size < 6:  # fewer than three points cannot enclose an area
            continue
        xs = coords[0::2]
        ys = coords[1::2]
        rows, cols = draw_polygon(ys, xs, shape=(height, width))
        mask[rows, cols] = True
    return mask


def rle_to_mask(rle: dict[str, Any], height: int, width: int) -> np.ndarray:
    """Decode a COCO RLE segmentation (uncompressed counts) into a mask."""
    counts = rle.get("counts")
    size = rle.get("size", [height, width])
    if isinstance(counts, (str, bytes)):
        try:
            from pycocotools import mask as mask_utils  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "This annotation uses compressed RLE, which needs pycocotools. "
                "Install it with 'pip install pycocotools'."
            ) from exc
        return np.asarray(mask_utils.decode(rle), dtype=bool)

    flat = np.zeros(int(size[0]) * int(size[1]), dtype=bool)
    position = 0
    value = False
    for run in counts:
        run = int(run)
        if value:
            flat[position : position + run] = True
        position += run
        value = not value
    # COCO RLE is column-major.
    return flat.reshape((int(size[1]), int(size[0]))).T


def decode_segmentation(segmentation: Any, height: int, width: int) -> np.ndarray:
    """Turn any supported COCO segmentation representation into a boolean mask."""
    if segmentation is None:
        return np.zeros((height, width), dtype=bool)
    if isinstance(segmentation, dict):
        return rle_to_mask(segmentation, height, width)
    if isinstance(segmentation, (list, tuple)):
        if not segmentation:
            return np.zeros((height, width), dtype=bool)
        first = segmentation[0]
        if isinstance(first, (list, tuple, np.ndarray)):
            return polygons_to_mask(segmentation, height, width)
        # A bare flat list is a single polygon.
        return polygons_to_mask([segmentation], height, width)
    raise TypeError(f"unsupported segmentation type: {type(segmentation)!r}")


def _parse_spine(raw: Any) -> np.ndarray | None:
    """Normalise a spine annotation to an ``(N, 2)`` array of ``(y, x)`` points.

    Spines appear either as a flat ``[x0, y0, x1, y1, ...]`` list, as a list of
    ``[x, y]`` pairs, or wrapped in an extra list (COCO's multi-part style).
    Keypoint-style triples ``[x, y, visibility]`` are handled too.
    """
    if raw is None:
        return None
    array = np.asarray(raw, dtype=np.float64)
    if array.size == 0:
        return None

    # Unwrap a single extra level of nesting, e.g. [[x0, y0, x1, y1, ...]].
    while array.ndim > 2:
        array = array[0]
    if array.ndim == 1:
        if array.size % 3 == 0 and array.size % 2 != 0:
            pairs = array.reshape(-1, 3)[:, :2]
        elif array.size % 2 == 0:
            pairs = array.reshape(-1, 2)
        else:
            return None
    elif array.ndim == 2:
        if array.shape[1] >= 2:
            pairs = array[:, :2]
        else:
            return None
    else:
        return None

    if pairs.shape[0] < 2:
        return None
    # Stored as (x, y); the rest of this codebase works in (row, column).
    return np.stack([pairs[:, 1], pairs[:, 0]], axis=1).astype(np.float32)


def _parse_chirality(raw: Any) -> int:
    """Normalise a chirality label to 0 (unknown), 1 (sinistral) or 2 (dextral)."""
    if raw is None:
        return 0
    if isinstance(raw, (int, np.integer)):
        return int(raw) if 0 <= int(raw) <= 2 else 0
    if isinstance(raw, float):
        return int(raw) if 0 <= int(raw) <= 2 else 0
    text = str(raw).strip().lower()
    if text in ("1", "left", "sinistral", "l"):
        return 1
    if text in ("2", "right", "dextral", "r"):
        return 2
    return 0


def _first_present(record: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, [], ""):
            return record[key]
    return None


def load_coco(path: str | Path) -> tuple[list[ImageRecord], dict[str, Any]]:
    """Load a MAGFiLO-style COCO annotation file.

    Args:
        path: Path to the annotation JSON.

    Returns:
        ``(records, meta)`` where ``records`` is a list of :class:`ImageRecord`
        in the order the images appear in the file, and ``meta`` holds the
        ``info``, ``licenses`` and ``categories`` blocks.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)

    meta = {key: raw.get(key) for key in ("info", "licenses", "categories")}

    # Build the category -> chirality map before reading annotations.
    category_chirality: dict[ImageId, int] = {}
    for category in raw.get("categories") or []:
        name = str(category.get("name", "")).strip().lower()
        if name in CHIRALITY_FROM_CATEGORY:
            category_chirality[normalise_id(category.get("id"))] = (
                CHIRALITY_FROM_CATEGORY[name]
            )

    records: dict[ImageId, ImageRecord] = {}
    for entry in raw.get("images", []):
        image_id = normalise_id(entry["id"])
        known = {"id", "file_name", "height", "width"}
        records[image_id] = ImageRecord(
            image_id=image_id,
            file_name=str(entry.get("file_name", f"{image_id}.fits")),
            height=int(entry.get("height", 0)),
            width=int(entry.get("width", 0)),
            extra={k: v for k, v in entry.items() if k not in known},
        )

    skipped = 0
    for entry in raw.get("annotations", []):
        image_id = normalise_id(entry["image_id"])
        record = records.get(image_id)
        if record is None:
            skipped += 1
            continue
        bbox = entry.get("bbox") or [0.0, 0.0, 0.0, 0.0]
        category_id = normalise_id(entry.get("category_id", 1))
        chirality = _parse_chirality(_first_present(entry, CHIRALITY_KEYS))
        if chirality == 0:
            # Fall back to the category, which is where MAGFiLO keeps it.
            chirality = category_chirality.get(category_id, 0)
        record.annotations.append(
            FilamentAnnotation(
                annotation_id=normalise_id(
                    entry.get("id", len(record.annotations) + 1)
                ),
                image_id=image_id,
                bbox=tuple(float(v) for v in bbox[:4]),  # type: ignore[arg-type]
                segmentation=entry.get("segmentation"),
                area=float(entry.get("area", 0.0)),
                spine=_parse_spine(_first_present(entry, SPINE_KEYS)),
                chirality=chirality,
                category_id=category_id,
            )
        )

    if skipped:
        import warnings

        warnings.warn(
            f"{skipped} annotation(s) referenced an image id not present in the file",
            stacklevel=2,
        )

    return list(records.values()), meta


def summarise(records: Sequence[ImageRecord]) -> dict[str, Any]:
    """Compute a few headline statistics, useful for a sanity check after loading."""
    counts = [len(record.annotations) for record in records]
    chirality: dict[str, int] = {name: 0 for name in CHIRALITY_NAMES.values()}
    with_spine = 0
    for record in records:
        for annotation in record.annotations:
            chirality[CHIRALITY_NAMES.get(annotation.chirality, "unknown")] += 1
            if annotation.spine is not None:
                with_spine += 1
    return {
        "n_images": len(records),
        "n_filaments": int(sum(counts)),
        "filaments_per_image_mean": float(np.mean(counts)) if counts else 0.0,
        "filaments_per_image_max": int(max(counts)) if counts else 0,
        "images_without_filaments": int(sum(1 for c in counts if c == 0)),
        "annotations_with_spine": with_spine,
        "chirality": chirality,
    }


def mask_to_polygons(
    mask: np.ndarray, tolerance: float = 1.0, min_points: int = 3
) -> list[list[float]]:
    """Trace a boolean mask to COCO-style polygons ``[x0, y0, x1, y1, ...]``.

    This is the inverse of :func:`polygons_to_mask` and is used when writing
    predictions back out in COCO form.

    Args:
        mask: Boolean mask to trace.
        tolerance: Douglas-Peucker simplification tolerance in pixels.  Larger
            values give smaller files at the cost of clipping fine detail, so
            keep it at or below one pixel for filament barbs.
        min_points: Discard rings with fewer vertices than this.
    """
    from skimage.measure import approximate_polygon, find_contours

    # Pad so that regions touching the border still produce a closed contour.
    padded = np.pad(mask.astype(np.uint8), 1, mode="constant")
    polygons: list[list[float]] = []
    for contour in find_contours(padded, 0.5):
        contour = contour - 1.0  # undo the padding offset
        if tolerance > 0:
            contour = approximate_polygon(contour, tolerance)
        if len(contour) < min_points:
            continue
        # find_contours returns (row, column); COCO wants (x, y).
        flat = np.stack([contour[:, 1], contour[:, 0]], axis=1).ravel()
        polygons.append([round(float(v), 2) for v in flat])
    return polygons


def mask_to_rle(mask: np.ndarray) -> dict[str, Any]:
    """Encode a boolean mask as an uncompressed, column-major COCO RLE."""
    flat = np.asarray(mask, dtype=bool).T.ravel()
    # Runs always start with a background run, which may have length zero.
    changes = np.flatnonzero(np.diff(flat)) + 1
    boundaries = np.concatenate(([0], changes, [flat.size]))
    counts = np.diff(boundaries).tolist()
    if flat.size and flat[0]:
        counts = [0] + counts
    return {"size": [int(mask.shape[0]), int(mask.shape[1])], "counts": [int(c) for c in counts]}


def _scale_segmentation(segmentation: Any, scale_x: float, scale_y: float) -> Any:
    """Scale a polygon segmentation. RLE cannot be scaled and is returned as is."""
    if segmentation is None or isinstance(segmentation, dict):
        return segmentation
    if not isinstance(segmentation, (list, tuple)) or not segmentation:
        return segmentation

    def scale_ring(ring: Sequence[float]) -> list[float]:
        coords = np.asarray(ring, dtype=np.float64).ravel()
        coords[0::2] *= scale_x
        coords[1::2] *= scale_y
        return coords.tolist()

    first = segmentation[0]
    if isinstance(first, (list, tuple, np.ndarray)):
        return [scale_ring(ring) for ring in segmentation]
    return [scale_ring(segmentation)]


def rescale_record(record: ImageRecord, height: int, width: int) -> ImageRecord:
    """Rescale an annotation record to a different image size, in place.

    Distributed image sets are sometimes resized relative to the frames the
    annotations were drawn on -- a JPEG release downsampled from the original
    FITS, for instance.  Silently ignoring that mismatch produces masks that are
    subtly offset and scaled, which is very hard to spot and destroys training.
    Rescaling the annotations to match the image keeps the image at its native
    resolution, which matters because barbs do not survive downsampling.

    Args:
        record: The record to modify.
        height: Height of the actual image, in pixels.
        width: Width of the actual image, in pixels.

    Returns:
        The same record, modified in place.
    """
    if record.height == height and record.width == width:
        return record
    if record.height <= 0 or record.width <= 0:
        record.height, record.width = height, width
        return record

    scale_y = height / record.height
    scale_x = width / record.width

    for annotation in record.annotations:
        x, y, w, h = annotation.bbox
        annotation.bbox = (x * scale_x, y * scale_y, w * scale_x, h * scale_y)
        annotation.segmentation = _scale_segmentation(
            annotation.segmentation, scale_x, scale_y
        )
        annotation.area *= scale_x * scale_y
        if annotation.spine is not None:
            spine = annotation.spine.astype(np.float64)
            spine[:, 0] *= scale_y  # rows
            spine[:, 1] *= scale_x  # columns
            annotation.spine = spine.astype(np.float32)

    record.height, record.width = height, width
    return record
