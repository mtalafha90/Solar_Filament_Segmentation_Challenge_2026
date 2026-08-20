"""Writing predictions out in the formats a segmentation challenge expects.

Three formats are supported, because competitions differ and the organisers'
exact requirement may change:

* **COCO instance JSON** -- one entry per predicted filament, with a polygon or
  RLE segmentation.  This is the natural match for MAGFiLO's own format and is
  what preserves the instance structure the challenge asks for.
* **Run-length encoded CSV** -- the common Kaggle layout, one row per instance.
* **PNG label maps** -- for inspection and for any pipeline that wants images.

Always check the competition's submission page and pick the matching writer.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from .data.coco import mask_to_polygons, mask_to_rle


def kaggle_rle(mask: np.ndarray) -> str:
    """Encode a mask in the run-length format Kaggle competitions normally use.

    The convention is one-indexed, ordered down columns first, and written as
    space-separated ``start length`` pairs.
    """
    pixels = np.asarray(mask, dtype=bool).T.ravel()
    if not pixels.any():
        return ""
    padded = np.concatenate(([False], pixels, [False]))
    changes = np.flatnonzero(padded[1:] != padded[:-1]) + 1
    starts = changes[0::2]
    ends = changes[1::2]
    lengths = ends - starts
    return " ".join(f"{int(s)} {int(l)}" for s, l in zip(starts, lengths))


def decode_kaggle_rle(encoded: str, height: int, width: int) -> np.ndarray:
    """Decode :func:`kaggle_rle` back into a boolean mask."""
    mask = np.zeros(height * width, dtype=bool)
    if encoded and encoded.strip():
        values = [int(v) for v in encoded.split()]
        for start, length in zip(values[0::2], values[1::2]):
            mask[start - 1 : start - 1 + length] = True
    return mask.reshape((width, height)).T


def instances_from_labels(
    labels: np.ndarray, probability: np.ndarray | None = None
) -> list[tuple[np.ndarray, float]]:
    """Split a label map into ``(mask, score)`` pairs.

    The score is the mean predicted probability inside the instance, which is
    what a detection metric ranks by.  Without a probability map every instance
    scores 1.0.
    """
    out: list[tuple[np.ndarray, float]] = []
    for value in np.unique(labels):
        if value <= 0:
            continue
        mask = labels == value
        score = float(probability[mask].mean()) if probability is not None else 1.0
        out.append((mask, score))
    return out


def write_coco(
    path: str | Path,
    predictions: Iterable[tuple[int, np.ndarray, np.ndarray | None]],
    use_rle: bool = False,
    polygon_tolerance: float = 0.8,
) -> int:
    """Write predictions as a list of COCO instance results.

    Args:
        path: Output JSON path.
        predictions: Iterable of ``(image_id, label_map, probability_map)``.
        use_rle: Write RLE segmentations instead of polygons.  RLE is exact;
            polygons are smaller but lose sub-pixel detail, so keep the
            tolerance low if fine structure is being scored.
        polygon_tolerance: Simplification tolerance in pixels for polygons.

    Returns:
        The number of instances written.
    """
    results: list[dict] = []
    for image_id, labels, probability in predictions:
        for mask, score in instances_from_labels(labels, probability):
            rows, cols = np.nonzero(mask)
            if rows.size == 0:
                continue
            if use_rle:
                segmentation: object = mask_to_rle(mask)
            else:
                polygons = mask_to_polygons(mask, tolerance=polygon_tolerance)
                if not polygons:
                    continue
                segmentation = polygons
            results.append(
                {
                    # Kept exactly as the dataset gave it. MAGFiLO ids are the
                    # original GONG frame names, not integers, and a grader
                    # matching on them needs them back unchanged.
                    "image_id": image_id,
                    "category_id": 1,
                    "segmentation": segmentation,
                    "bbox": [
                        float(cols.min()),
                        float(rows.min()),
                        float(cols.max() - cols.min() + 1),
                        float(rows.max() - rows.min() + 1),
                    ],
                    "area": float(mask.sum()),
                    "score": round(score, 5),
                }
            )

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(results, handle)
    return len(results)


def write_rle_csv(
    path: str | Path,
    predictions: Iterable[tuple[str | int, np.ndarray, np.ndarray | None]],
    id_column: str = "ImageId",
    rle_column: str = "EncodedPixels",
    include_empty: bool = True,
) -> int:
    """Write predictions as one run-length encoded row per instance.

    Args:
        path: Output CSV path.
        predictions: Iterable of ``(image_id, label_map, probability_map)``.
        id_column: Name of the identifier column.
        rle_column: Name of the encoded-mask column.
        include_empty: Emit a blank row for images with no detections.  Most
            graders require every test image to appear exactly once, so leaving
            this on is usually the safe choice.

    Returns:
        The number of rows written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([id_column, rle_column])
        for image_id, labels, _ in predictions:
            instances = instances_from_labels(labels)
            if not instances and include_empty:
                writer.writerow([image_id, ""])
                rows += 1
                continue
            for mask, _score in instances:
                writer.writerow([image_id, kaggle_rle(mask)])
                rows += 1
    return rows


def write_label_pngs(
    directory: str | Path,
    predictions: Iterable[tuple[str, np.ndarray]],
) -> int:
    """Write each label map as a 16-bit PNG, for inspection or downstream tools."""
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    count = 0
    for name, labels in predictions:
        array = np.asarray(labels, dtype=np.uint16)
        Image.fromarray(array, mode="I;16").save(directory / f"{Path(name).stem}.png")
        count += 1
    return count


def summarise_predictions(labels_list: Sequence[np.ndarray]) -> dict[str, float]:
    """Headline statistics for a set of predictions, as a submission sanity check.

    Worth glancing at before uploading: a filament count far from the expected
    six or seven per frame, or a coverage far from a per cent or two, usually
    means the threshold is wrong.
    """
    counts = [int(labels.max()) for labels in labels_list]
    coverage = [float((labels > 0).mean()) for labels in labels_list]
    areas: list[float] = []
    for labels in labels_list:
        for value in np.unique(labels):
            if value > 0:
                areas.append(float(np.count_nonzero(labels == value)))
    return {
        "n_images": len(labels_list),
        "total_instances": int(sum(counts)),
        "instances_per_image_mean": float(np.mean(counts)) if counts else 0.0,
        "instances_per_image_max": int(max(counts)) if counts else 0,
        "images_with_no_detection": int(sum(1 for c in counts if c == 0)),
        "mean_pixel_coverage": float(np.mean(coverage)) if coverage else 0.0,
        "instance_area_median": float(np.median(areas)) if areas else 0.0,
        "instance_area_min": float(np.min(areas)) if areas else 0.0,
        "instance_area_max": float(np.max(areas)) if areas else 0.0,
    }


# --------------------------------------------------------------------------
# The Solar Filament Segmentation Challenge 2026 submission format
# --------------------------------------------------------------------------

CHALLENGE_COLUMNS = ("filament_id", "segmentation_rle")


def coco_rle_counts(mask: np.ndarray) -> str:
    """Encode a mask as pycocotools compressed-RLE counts.

    This is the exact string the challenge expects in ``segmentation_rle``: the
    ``counts`` field of a pycocotools RLE, decoded to ASCII, with the ``size``
    omitted because every frame is 2048 x 2048.

    The encoding is lossless and is the organisers' own, so a submission decodes
    back to precisely the mask that was predicted.

    Args:
        mask: Boolean mask to encode.

    Returns:
        The counts string, free of quotes and commas so it needs no escaping.

    Raises:
        ImportError: If pycocotools is not installed.
    """
    try:
        from pycocotools import mask as mask_utils
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "The challenge submission format needs pycocotools. "
            "Install it with 'pip install pycocotools'."
        ) from exc

    encoded = mask_utils.encode(np.asfortranarray(np.asarray(mask, dtype=np.uint8)))
    counts = encoded["counts"]
    return counts.decode("ascii") if isinstance(counts, bytes) else str(counts)


def decode_coco_rle_counts(counts: str, height: int = 2048, width: int = 2048) -> np.ndarray:
    """Decode challenge ``segmentation_rle`` counts back into a mask.

    The inverse of :func:`coco_rle_counts`, used to check a submission before
    uploading it.
    """
    from pycocotools import mask as mask_utils

    rle = {"size": [int(height), int(width)], "counts": counts.encode("ascii")}
    return np.asarray(mask_utils.decode(rle), dtype=bool)


def image_id_from_name(file_name: str) -> str:
    """The observation id the challenge expects, taken from an image file name.

    Test images are named ``YYYYMMDDHHMMSSII.jpeg`` -- capture time and
    instrument -- and the submission keys on that stem. Any annotator prefix
    that a *training* id carries is stripped, since predictions are made on the
    image, not on one annotator's reading of it.
    """
    stem = Path(str(file_name)).name
    for suffix in (".gz", ".fz"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
    stem = Path(stem).stem
    # A training id like '010401-20160920230134Lh' carries an annotator batch.
    if "-" in stem:
        head, _, tail = stem.partition("-")
        if head.isdigit() and tail:
            stem = tail
    return stem


def write_challenge_csv(
    path: str | Path,
    predictions: Iterable[tuple[str, np.ndarray]],
    expected_shape: tuple[int, int] = (2048, 2048),
) -> int:
    """Write the competition's submission file.

    One row per predicted filament::

        filament_id,segmentation_rle
        20150125172714Mh_1,^Vj02jo16I5O2O1`PNA]o1c0N19G1N11O3L01O4JYamT3
        20150125172714Mh_2,...

    Images with no detected filament contribute no rows, which is correct: the
    evaluation matches predictions to ground truth by overlap, not by index, so
    a blank row would register as a spurious empty segment.

    Args:
        path: Output CSV path.
        predictions: Iterable of ``(image_name_or_id, label_map)``.
        expected_shape: Frame size the challenge assumes. A mismatch is raised
            rather than written, because the size is omitted from the encoding
            and a wrong one would decode into nonsense on the organisers' side.

    Returns:
        The number of filament rows written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(CHALLENGE_COLUMNS)
        for name, labels in predictions:
            labels = np.asarray(labels)
            if labels.shape != tuple(expected_shape):
                raise ValueError(
                    f"{name}: mask is {labels.shape}, but the challenge fixes the "
                    f"frame at {tuple(expected_shape)} and the submission omits the "
                    f"size, so this would decode incorrectly."
                )
            image_id = image_id_from_name(name)
            for number, value in enumerate(
                (v for v in np.unique(labels) if v > 0), start=1
            ):
                writer.writerow(
                    [f"{image_id}_{number}", coco_rle_counts(labels == value)]
                )
                rows += 1
    return rows


def read_challenge_csv(
    path: str | Path, shape: tuple[int, int] = (2048, 2048)
) -> dict[str, list[np.ndarray]]:
    """Read a submission back into masks per image, for checking it.

    Returns:
        A mapping from image id to the list of predicted masks.
    """
    out: dict[str, list[np.ndarray]] = {}
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            filament_id = row[CHALLENGE_COLUMNS[0]]
            image_id = filament_id.rsplit("_", 1)[0]
            mask = decode_coco_rle_counts(row[CHALLENGE_COLUMNS[1]], *shape)
            out.setdefault(image_id, []).append(mask)
    return out
