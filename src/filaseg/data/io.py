"""Reading full-disk solar images from the formats the challenge data uses.

GONG distributes H-alpha observations as FITS.  Derived releases and Kaggle
mirrors often ship the same frames as 8-bit PNG or JPEG.  Everything here
returns a single-channel ``float32`` array so the rest of the pipeline does not
have to care which it was given.
"""

from __future__ import annotations

from pathlib import Path

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


def find_image(directory: str | Path, file_name: str) -> Path:
    """Locate an image referenced by a COCO ``file_name``, tolerating format swaps.

    Annotation files often name a ``.fits`` frame while the distributed images
    are ``.jpg``, or sit one directory deeper.  This looks for the exact name
    first, then the same stem with any known image extension, then anywhere
    below the directory.
    """
    directory = Path(directory)
    direct = directory / file_name
    if direct.exists():
        return direct

    stem = Path(file_name).stem
    # A FITS name like 'foo.fits' has stem 'foo'; 'foo.fits.gz' needs a second strip.
    if stem.endswith(".fits"):
        stem = stem[: -len(".fits")]

    for suffix in (*FITS_SUFFIXES, *IMAGE_SUFFIXES, ".npy"):
        candidate = directory / f"{stem}{suffix}"
        if candidate.exists():
            return candidate

    matches = sorted(directory.rglob(f"{stem}.*"))
    for match in matches:
        if match.suffix.lower() in FITS_SUFFIXES | IMAGE_SUFFIXES | {".npy"}:
            return match

    raise FileNotFoundError(f"no image for '{file_name}' under {directory}")
