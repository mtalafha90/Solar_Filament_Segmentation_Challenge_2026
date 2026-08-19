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
    directory: str | Path, file_name: str, search_subdirectories: bool = False
) -> Path:
    """Locate an image referenced by a COCO ``file_name``.

    Annotation files often name a ``.fits`` frame while the distributed images
    are ``.jpeg``, and the ``file_name`` may carry a directory prefix that does
    not match how the data was unpacked.  Both are handled by matching on the
    stem.

    A recursive search is *not* done by default.  It sounds helpful and is
    dangerous: if a record's image is genuinely absent -- because the annotation
    file covers observations that were not distributed, which is normal for a
    competition split -- a loose search can match some other file, and the
    record is then trained on the wrong image with no error raised. Silently
    pairing a mask with the wrong frame is far worse than a missing file.

    Args:
        directory: Where the images live.
        file_name: The ``file_name`` field from the annotations.
        search_subdirectories: Also search below ``directory``, matching the
            stem exactly. Only enable this when the images really are nested.

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
            if candidate.suffix.lower() in known and candidate.stem == stem:
                return candidate

    raise FileNotFoundError(f"no image for '{file_name}' under {directory}")


def _split_prefix(file_name: str) -> str:
    """The directory a ``file_name`` claims to sit in, lower-cased. May be empty."""
    parent = Path(file_name).parent
    return "" if str(parent) in (".", "") else parent.name.lower()


def resolve_images(
    directory: str | Path,
    file_names: Sequence[str],
    search_subdirectories: bool = False,
) -> tuple[dict[str, Path], list[str], dict[Path, list[str]]]:
    """Resolve many ``file_name`` values at once and report what went wrong.

    Two records resolving to the same file means annotations would be paired
    with the wrong frame, which no downstream step could detect. That usually
    happens when one annotation file covers several splits and the ``file_name``
    fields carry a directory prefix -- ``train/x.jpeg`` and ``test/x.jpeg`` --
    that is lost when matching on the stem. Where the prefixes make the intent
    clear, we resolve it by keeping only the records whose prefix matches the
    directory being read; otherwise the collision is reported.

    Args:
        directory: Where the images live.
        file_names: The ``file_name`` fields to resolve.
        search_subdirectories: Passed through to :func:`find_image`.

    Returns:
        ``(resolved, missing, collisions)``.  ``resolved`` maps each file name
        that was found to its path; ``missing`` lists those that were not;
        ``collisions`` maps any path still claimed by more than one name to the
        names claiming it. A non-empty ``collisions`` should always be treated
        as an error.
    """
    directory = Path(directory)

    # If the names carry directory prefixes and one of them matches this
    # directory, the others belong to a different split. Drop them up front.
    target = directory.name.lower()
    prefixes = {_split_prefix(name) for name in file_names}
    if target in prefixes and len(prefixes - {""}) > 1:
        file_names = [
            name for name in file_names if _split_prefix(name) in (target, "")
        ]

    resolved: dict[str, Path] = {}
    missing: list[str] = []
    claimed: dict[Path, list[str]] = {}

    for name in file_names:
        try:
            path = find_image(directory, name, search_subdirectories)
        except FileNotFoundError:
            missing.append(name)
            continue
        resolved[name] = path
        claimed.setdefault(path.resolve(), []).append(name)

    collisions = {path: names for path, names in claimed.items() if len(names) > 1}
    return resolved, missing, collisions
