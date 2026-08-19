"""Finding the pieces of a dataset on disk.

Competition downloads are laid out predictably -- images under ``train/`` and
``test/``, with one annotation JSON alongside the training images -- but the
annotation file is never called the same thing twice.  MAGFiLO's is
``MAGFiLO_1.0_Annotations_kaggle2026_train.json``.  Rather than make every
caller know that, the helpers here find it.

The important behaviour is in :func:`resolve_annotations`: when it is handed a
path that does not exist, it looks for the real file next to it and says exactly
what it found, instead of failing with a bare ``FileNotFoundError`` about a name
the user only typed because an example told them to.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

from .io import FITS_SUFFIXES, IMAGE_SUFFIXES

DATA_SUFFIXES = FITS_SUFFIXES | IMAGE_SUFFIXES | {".npy"}


@dataclass
class DatasetLayout:
    """Where the parts of a dataset live."""

    annotations: Path | None = None
    train_dir: Path | None = None
    test_dir: Path | None = None

    def describe(self) -> str:
        return (
            f"annotations: {self.annotations}\n"
            f"train images: {self.train_dir}\n"
            f"test images: {self.test_dir}"
        )


def find_annotation_files(directory: Path) -> list[Path]:
    """List candidate annotation JSONs in a directory and one level below it.

    Sorted largest first, because the real annotation file dwarfs anything else
    that might be sitting beside it.
    """
    directory = Path(directory)
    if not directory.is_dir():
        return []
    candidates = sorted(directory.glob("*.json")) + sorted(directory.glob("*/*.json"))
    unique = {path.resolve(): path for path in candidates}
    return sorted(unique.values(), key=lambda p: p.stat().st_size, reverse=True)


def count_images(directory: Path | None) -> list[Path]:
    """Every readable image below ``directory``."""
    if directory is None or not Path(directory).is_dir():
        return []
    return sorted(
        p for p in Path(directory).rglob("*") if p.suffix.lower() in DATA_SUFFIXES
    )


def discover(data_dir: str | Path) -> DatasetLayout:
    """Work out a dataset's layout from its root directory.

    Handles ``<root>/train`` plus ``<root>/test``, and the flat case where the
    images sit directly in the root.
    """
    data_dir = Path(data_dir)
    train_dir = next(
        (
            d
            for d in (data_dir / "train", data_dir / "training", data_dir)
            if d.is_dir() and count_images(d)
        ),
        None,
    )
    test_dir = next(
        (d for d in (data_dir / "test", data_dir / "testing") if d.is_dir()), None
    )

    annotations = None
    for candidate_dir in [d for d in (train_dir, data_dir) if d is not None]:
        found = find_annotation_files(candidate_dir)
        if found:
            annotations = found[0]
            break
    return DatasetLayout(annotations, train_dir, test_dir)


def resolve_annotations(
    annotations: str | Path | None = None,
    image_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
) -> Path:
    """Find the annotation JSON, being forgiving about the exact name.

    Tried in order: the path given, if it exists; any JSON sitting beside that
    path; any JSON in ``image_dir``; and finally a full discovery pass over
    ``data_dir``.

    Args:
        annotations: The path the caller asked for. May be wrong or missing.
        image_dir: Directory holding the images, often next to the JSON.
        data_dir: Dataset root, searched last.

    Returns:
        Path to the annotation file.

    Raises:
        FileNotFoundError: With the candidates that *were* found, so the caller
            can see what to use instead.
    """
    if annotations is not None:
        given = Path(annotations)
        if given.exists():
            return given

        # The name is wrong. Look for the real thing next to where it was
        # expected, and say so rather than reporting the name back as missing.
        for directory in [d for d in (given.parent, Path(image_dir) if image_dir else None) if d]:
            found = find_annotation_files(directory)
            if len(found) == 1:
                warnings.warn(
                    f"'{given}' does not exist; using '{found[0]}' instead",
                    stacklevel=2,
                )
                return found[0]
            if len(found) > 1:
                listing = "\n  ".join(str(p) for p in found)
                raise FileNotFoundError(
                    f"'{given}' does not exist, and there is more than one JSON in "
                    f"{directory}. Pass --annotations with one of:\n  {listing}"
                )

    for directory in [Path(image_dir)] if image_dir else []:
        found = find_annotation_files(directory)
        if found:
            return found[0]

    if data_dir is not None:
        layout = discover(data_dir)
        if layout.annotations is not None:
            return layout.annotations

    searched = [str(p) for p in (annotations, image_dir, data_dir) if p is not None]
    raise FileNotFoundError(
        "No annotation JSON found. Looked in: "
        + (", ".join(searched) or "nowhere -- pass --annotations or --data-dir")
    )
