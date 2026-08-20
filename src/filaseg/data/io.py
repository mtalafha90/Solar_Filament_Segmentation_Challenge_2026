"""Reading full-disk solar images from the formats the challenge data uses.

GONG distributes H-alpha observations as FITS.  Derived releases and Kaggle
mirrors often ship the same frames as 8-bit PNG or JPEG.  Everything here
returns a single-channel ``float32`` array so the rest of the pipeline does not
have to care which it was given.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np

FITS_SUFFIXES = {".fits", ".fit", ".fts", ".fz"}
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def _read_fits(path: Path) -> np.ndarray:
    try:
        from astropy.io import fits
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading FITS files needs astropy. Install it with 'pip install astropy'."
        ) from exc

    with fits.open(path, memmap=False) as hdul:
        for hdu in hdul:
            data = getattr(hdu, "data", None)
            if data is not None and np.ndim(data) >= 2:
                array = np.asarray(data, dtype=np.float32)
                # Some archives store a trivial leading axis.
                while array.ndim > 2 and array.shape[0] == 1:
                    array = array[0]
                if array.ndim != 2:
                    raise ValueError(f"{path}: expected 2-D image data, got {array.shape}")
                return array
    raise ValueError(f"{path}: no image data found in any HDU")


def _read_bitmap(path: Path) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "Reading PNG/JPEG files needs Pillow. Install it with 'pip install pillow'."
        ) from exc

    with Image.open(path) as handle:
        if handle.mode not in ("I;16", "I", "F", "L"):
            handle = handle.convert("L")
        array = np.asarray(handle, dtype=np.float32)
    if array.ndim == 3:
        array = array.mean(axis=2)
    return array


def read_image(path: str | Path) -> np.ndarray:
    """Read a full-disk image from FITS, PNG, JPEG or TIFF.

    Args:
        path: File to read.

    Returns:
        A 2-D ``float32`` array.  Values are left on their native scale; the
        preprocessing chain normalises them.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    suffix = path.suffix.lower()
    if suffix in FITS_SUFFIXES or path.name.lower().endswith(".fits.gz"):
        return _read_fits(path)
    if suffix in IMAGE_SUFFIXES:
        return _read_bitmap(path)
    if suffix == ".npy":
        array = np.load(path).astype(np.float32)
        if array.ndim != 2:
            raise ValueError(f"{path}: expected 2-D array, got {array.shape}")
        return array
    # Unknown extension: try FITS first, then a bitmap.
    try:
        return _read_fits(path)
    except Exception:
        return _read_bitmap(path)


def find_image(
    directory: str | Path, file_name: str, search_subdirectories: bool = True
) -> Path:
    """Locate an image referenced by a COCO ``file_name``.

    Annotation files often name a ``.fits`` frame while the distributed images
    are ``.jpeg``, and the ``file_name`` may carry a directory prefix that does
    not match how the data was unpacked.  Both are handled by matching on the
    stem.

    Subdirectories are searched, because releases often nest the frames one
    level below the split directory. The match is on the *exact* stem, never a
    wildcard: a record whose image was not distributed must come back missing
    rather than quietly latching on to some other frame, since pairing a mask
    with the wrong image is far worse than a missing file.

    Args:
        directory: Where the images live.
        file_name: The ``file_name`` field from the annotations.
        search_subdirectories: Also search below ``directory``, matching the
            stem exactly.

    Returns:
        Path to the image.

    Raises:
        FileNotFoundError: If no image matches.
    """
    directory = Path(directory)
    direct = directory / file_name
    if direct.exists():
        return direct

    # Strip any directory prefix the annotations carry, then any FITS suffix.
    stem = Path(file_name).name
    for suffix in (".gz", ".fz"):
        if stem.lower().endswith(suffix):
            stem = stem[: -len(suffix)]
    stem = Path(stem).stem

    known = FITS_SUFFIXES | IMAGE_SUFFIXES | {".npy"}
    for suffix in (*IMAGE_SUFFIXES, *FITS_SUFFIXES, ".npy"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    if search_subdirectories:
        for candidate in sorted(directory.rglob("*")):
            if not candidate.is_file() or candidate.suffix.lower() not in known:
                continue
            candidate_stem = candidate.stem
            if candidate_stem.lower().endswith(".fits"):
                candidate_stem = candidate_stem[: -len(".fits")]
            if candidate_stem == stem:
                return candidate

    raise FileNotFoundError(f"no image for '{file_name}' under {directory}")


def _split_prefix(file_name: str) -> str:
    """The directory a ``file_name`` claims to sit in, lower-cased. May be empty."""
    parent = Path(file_name).parent
    return "" if str(parent) in (".", "") else parent.name.lower()


def build_image_index(directory: str | Path) -> dict[str, list[Path]]:
    """Map every image stem below ``directory`` to the files that carry it.

    Built once and reused, because resolving thousands of records by walking the
    tree each time is quadratic. Nested layouts are handled for free: images may
    sit directly in ``directory`` or in any subdirectory beneath it.
    """
    directory = Path(directory)
    known = FITS_SUFFIXES | IMAGE_SUFFIXES | {".npy"}
    index: dict[str, list[Path]] = {}
    if not directory.is_dir():
        return index
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in known:
            continue
        stem = path.stem
        if stem.lower().endswith(".fits"):
            stem = stem[: -len(".fits")]
        index.setdefault(stem, []).append(path)
    return index


def _stem_of(file_name: str) -> str:
    """The bare stem of a COCO ``file_name``, ignoring any directory prefix."""
    name = Path(file_name).name
    for suffix in (".gz", ".fz"):
        if name.lower().endswith(suffix):
            name = name[: -len(suffix)]
    return Path(name).stem


def resolve_images(
    directory: str | Path,
    file_names: Sequence[str],
    search_subdirectories: bool = True,
) -> tuple[dict[str, Path], list[str], dict[Path, list[str]]]:
    """Resolve many ``file_name`` values at once and report what went wrong.

    Matching is on the exact stem, never a wildcard: a record whose image was
    not distributed must come back missing rather than quietly latching on to
    some other frame. Competition splits routinely ship an annotation file
    covering more observations than the images beside it, so missing records are
    normal and are reported as a count.

    Two records resolving to the same file is not normal, and means annotations
    would be paired with the wrong frame. Where ``file_name`` values carry split
    prefixes -- ``train/x.jpeg`` against ``test/x.jpeg`` -- the two are
    separated automatically; anything left over is reported.

    Args:
        directory: Where the images live. Subdirectories are searched too, since
            releases often nest the frames one level down.
        file_names: The ``file_name`` fields to resolve.
        search_subdirectories: Search below ``directory``. On by default; the
            stem match is exact, so nesting adds no risk of a wrong match.

    Returns:
        ``(resolved, missing, collisions)``.
    """
    directory = Path(directory)

    # If the names carry directory prefixes and one matches this directory, the
    # others belong to a different split. Drop them before resolving.
    target = directory.name.lower()
    prefixes = {_split_prefix(name) for name in file_names}
    if target in prefixes and len(prefixes - {""}) > 1:
        file_names = [
            name for name in file_names if _split_prefix(name) in (target, "")
        ]

    index = build_image_index(directory) if search_subdirectories else {}

    resolved: dict[str, Path] = {}
    missing: list[str] = []
    claimed: dict[Path, list[str]] = {}

    for name in file_names:
        if name in resolved:
            continue  # the same file name listed twice resolves once
        path: Path | None = None

        direct = directory / name
        if direct.is_file():
            path = direct
        else:
            stem = _stem_of(name)
            for suffix in (*IMAGE_SUFFIXES, *FITS_SUFFIXES, ".npy"):
                candidate = directory / f"{stem}{suffix}"
                if candidate.is_file():
                    path = candidate
                    break
            if path is None:
                matches = index.get(stem)
                if matches:
                    path = matches[0]

        if path is None:
            missing.append(name)
            continue
        resolved[name] = path
        claimed.setdefault(path.resolve(), []).append(name)

    collisions = {path: names for path, names in claimed.items() if len(names) > 1}
    return resolved, missing, collisions
