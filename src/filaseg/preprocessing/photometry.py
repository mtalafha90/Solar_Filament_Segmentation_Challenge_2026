"""Photometric flattening of full-disk H-alpha images.

Two effects swamp the signal we actually care about:

1. **Limb darkening.** The Sun looks brighter at the centre of the disk than
   at the edge, by tens of per cent.  A filament near the limb is therefore
   darker in absolute terms than quiet Sun near disk centre, which makes any
   global intensity threshold useless and forces a network to waste capacity
   learning the radial ramp.
2. **Large-scale transmission gradients.** These are ground-based
   observations.  Thin cloud, haze and imperfect flat fields impose smooth,
   slowly varying brightness gradients that differ between the six GONG sites
   and between frames from the same site.

Both are removed by dividing the image by a smooth estimate of what the
brightness *would* be with no filaments present.  The estimators below are
deliberately robust: they use medians and morphological openings whose scale
is much larger than any filament width, so filaments survive the division
while the background does not.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .disk import SolarDisk, detect_disk

# Above this width, a Gaussian is evaluated on a decimated grid and interpolated
# back. A background estimated at sigma = 64 holds no detail finer than that, so
# computing it at full resolution is wasted work: the separable kernel grows
# with sigma, making the cost scale as sigma * N. On a 2048-pixel frame this is
# the difference between three seconds and a tenth of one, for an error far
# below the noise.
MAX_DIRECT_SIGMA = 8.0


def smooth_background(
    values: np.ndarray,
    weights: np.ndarray,
    sigma: float,
    max_direct_sigma: float = MAX_DIRECT_SIGMA,
) -> np.ndarray:
    """Normalised Gaussian convolution of ``values`` weighted by ``weights``.

    Dividing the smoothed values by the smoothed weights is what keeps the
    off-disk region from dragging the estimate down near the limb: pixels with
    zero weight contribute nothing rather than contributing zero.

    Wide kernels are evaluated on a decimated grid and interpolated back. The
    decimation is a box average, which is a proper anti-aliasing filter, and the
    result is smoothed far more than the decimation step, so nothing that
    survives the Gaussian is lost by taking it.
    """
    factor = 1
    if sigma > max_direct_sigma:
        factor = max(1, int(sigma // max_direct_sigma))

    if factor > 1:
        from skimage.measure import block_reduce

        small_values = block_reduce(values, (factor, factor), np.mean)
        small_weights = block_reduce(weights, (factor, factor), np.mean)
        scaled = sigma / factor
        numerator = ndi.gaussian_filter(small_values, scaled, mode="nearest")
        denominator = ndi.gaussian_filter(small_weights, scaled, mode="nearest")
        background = numerator / np.maximum(denominator, 1e-6)
        background = ndi.zoom(
            background,
            (values.shape[0] / background.shape[0], values.shape[1] / background.shape[1]),
            order=1,
            mode="nearest",
        )
        return np.asarray(background[: values.shape[0], : values.shape[1]], dtype=np.float32)

    numerator = ndi.gaussian_filter(values, sigma, mode="nearest")
    denominator = ndi.gaussian_filter(weights, sigma, mode="nearest")
    return (numerator / np.maximum(denominator, 1e-6)).astype(np.float32)


def radial_limb_profile(
    image: np.ndarray,
    disk: SolarDisk,
    n_bins: int = 256,
    max_radius: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Measure the median intensity as a function of distance from disk centre.

    Using the median (rather than the mean) is what makes this robust: filaments
    and sunspots are a small minority of pixels in any radial bin, so they move
    the median hardly at all.

    Returns:
        A pair ``(bin_centres, median_intensity)`` in units of solar radii.
    """
    radius_map = disk.radial_map(image.shape)
    on_disk = radius_map <= max_radius
    radii = radius_map[on_disk]
    values = image[on_disk]

    edges = np.linspace(0.0, max_radius, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    which = np.clip(np.digitize(radii, edges) - 1, 0, n_bins - 1)

    profile = np.full(n_bins, np.nan, dtype=np.float64)
    order = np.argsort(which, kind="stable")
    sorted_bins = which[order]
    sorted_values = values[order]
    starts = np.searchsorted(sorted_bins, np.arange(n_bins), side="left")
    ends = np.searchsorted(sorted_bins, np.arange(n_bins), side="right")
    for index, (start, end) in enumerate(zip(starts, ends)):
        if end > start:
            profile[index] = np.median(sorted_values[start:end])

    # Fill any empty bins (rare, but possible for tiny disks) by interpolation.
    good = np.isfinite(profile)
    if good.sum() < 2:
        profile = np.full(n_bins, float(np.median(values)) if values.size else 1.0)
    else:
        profile = np.interp(centres, centres[good], profile[good])
    return centres.astype(np.float32), profile.astype(np.float32)


def remove_limb_darkening(
    image: np.ndarray,
    disk: SolarDisk,
    n_bins: int = 256,
    smooth_bins: float = 3.0,
) -> np.ndarray:
    """Divide out the radial limb-darkening profile.

    The result is a "flattened" disk in which quiet Sun sits at roughly 1.0
    everywhere and filaments appear as consistent negative departures,
    regardless of where on the disk they happen to lie.
    """
    centres, profile = radial_limb_profile(image, disk, n_bins=n_bins)
    if smooth_bins > 0:
        profile = ndi.gaussian_filter1d(profile, smooth_bins, mode="nearest")

    radius_map = disk.radial_map(image.shape)
    # Beyond the limb the profile is meaningless; clamp to the outermost bin so
    # the division stays finite, and mask off-disk pixels separately.
    clamped = np.clip(radius_map, centres[0], centres[-1])
    background = np.interp(clamped, centres, profile)

    floor = max(1e-6, 0.02 * float(np.nanmedian(profile)))
    return (image / np.maximum(background, floor)).astype(np.float32)


def remove_large_scale_gradient(
    image: np.ndarray,
    valid: np.ndarray,
    scale: float = 64.0,
) -> np.ndarray:
    """Remove smooth transmission gradients while preserving filaments.

    ``scale`` is in pixels and must be comfortably larger than the width of any
    filament (filaments are typically a few to a few tens of pixels wide at
    GONG resolution).  Because the background is estimated at a much coarser
    scale than the structures of interest, dividing by it flattens haze and
    flat-field residuals without flattening the filaments themselves.

    Args:
        image: Image to flatten, usually already limb-darkening corrected.
        valid: Boolean mask of pixels to trust, normally the on-disk mask.
        scale: Smoothing length in pixels.
    """
    weights = valid.astype(np.float32)
    background = smooth_background(image * weights, weights, scale)

    floor = max(1e-6, 0.02 * float(np.nanmedian(background[valid])) if valid.any() else 1e-6)
    return (image / np.maximum(background, floor)).astype(np.float32)


def normalise(
    image: np.ndarray,
    valid: np.ndarray,
    low: float = 1.0,
    high: float = 99.0,
) -> np.ndarray:
    """Robustly rescale on-disk pixels to roughly [0, 1].

    Percentile clipping rather than min/max keeps a single hot pixel or a
    cosmic ray from compressing the whole dynamic range.
    """
    if not valid.any():
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(image[valid], [low, high])
    if not np.isfinite(hi - lo) or hi - lo < 1e-8:
        return np.zeros_like(image, dtype=np.float32)
    return np.clip((image - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def preprocess(
    image: np.ndarray,
    disk: SolarDisk | None = None,
    disk_fraction: float = 0.96,
    gradient_scale: float = 64.0,
    invert: bool = True,
) -> tuple[np.ndarray, np.ndarray, SolarDisk]:
    """Run the full preprocessing chain on a raw full-disk image.

    Steps: locate the limb, divide out limb darkening, divide out any smooth
    transmission gradient, mask everything outside the disk, normalise, and
    (by default) invert so that filaments are *bright*.  Inverting is a small
    convenience that makes filaments positive-valued features, which suits both
    the classical detector and the network's activation statistics.

    Args:
        image: Raw two-dimensional full-disk image.
        disk: Pre-computed disk geometry; detected automatically when omitted.
        disk_fraction: Fraction of the solar radius kept as valid.

            The default trims the outer four per cent, and that is not
            conservatism for its own sake. Real H-alpha observations carry a
            bright chromospheric rim just inside the limb, from spicules and
            emission that no flat field removes. Dividing out limb darkening
            leaves it as an enormous positive residual, and because the detector
            looks for bright departures from the local background, that rim
            outscores every genuine filament: measured on real GONG frames,
            88 per cent of the highest-scoring pixels fell in the outer three
            per cent of the disk, where 0.2 per cent of annotated filaments lie.
            The cost of trimming is small in comparison -- a couple of per cent
            of annotated filament pixels sit beyond 0.96 R, where they are
            severely foreshortened anyway.
        gradient_scale: Smoothing length for the transmission-gradient estimate.
        invert: Return ``1 - x`` on the disk, so filaments are bright.

    Returns:
        ``(processed, valid_mask, disk)``.  Pixels outside the disk are zero.
    """
    image = np.nan_to_num(np.asarray(image, dtype=np.float32), nan=0.0)
    if disk is None:
        disk = detect_disk(image)

    valid = disk.mask(image.shape, disk_fraction)
    flat = remove_limb_darkening(image, disk)
    flat = remove_large_scale_gradient(flat, valid, scale=gradient_scale)
    out = normalise(flat, valid)
    if invert:
        out = 1.0 - out
    return (out * valid).astype(np.float32), valid, disk
