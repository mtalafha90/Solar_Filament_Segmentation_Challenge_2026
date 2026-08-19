"""Solar disk geometry: finding the limb in a full-disk H-alpha image.

Every downstream step needs to know where the solar disk sits, because
filaments only exist on the disk and because the brightness of the disk
falls off towards its edge.  GONG images are already scaled to a nominal
solar radius of 900 pixels on a 2048x2048 grid, but the disk centre still
drifts between sites and observations, so we measure it rather than assume it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class SolarDisk:
    """Centre and radius of the solar disk, in pixels."""

    centre_y: float
    centre_x: float
    radius: float

    def radial_map(self, shape: tuple[int, int]) -> np.ndarray:
        """Distance of every pixel from disk centre, in units of solar radii."""
        yy, xx = np.ogrid[: shape[0], : shape[1]]
        dist = np.hypot(yy - self.centre_y, xx - self.centre_x)
        return (dist / self.radius).astype(np.float32)

    def mask(self, shape: tuple[int, int], fraction: float = 1.0) -> np.ndarray:
        """Boolean mask of pixels inside ``fraction`` of the solar radius."""
        return self.radial_map(shape) <= fraction


def _otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Otsu's threshold, implemented here to avoid a hard scikit-image dependency."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0
    hist, edges = np.histogram(finite, bins=bins)
    centres = 0.5 * (edges[:-1] + edges[1:])
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    valid = (weight1[:-1] > 0) & (weight2[1:] > 0)
    if not np.any(valid):
        return float(finite.mean())
    mean1 = np.cumsum(hist * centres) / np.maximum(weight1, 1)
    mean2 = (np.cumsum((hist * centres)[::-1]) / np.maximum(weight2[::-1], 1))[::-1]
    variance = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    variance = np.where(valid, variance, -np.inf)
    return float(centres[int(np.argmax(variance))])


def find_disk(image: np.ndarray, fill_holes: bool = True) -> SolarDisk:
    """Locate the solar disk in a full-disk image.

    The sky around the Sun is essentially black, so a single global threshold
    separates disk from sky very reliably.  We then take the largest connected
    bright region as the disk.  The radius is derived from the region's *area*
    rather than from its bounding box, because area is far less sensitive to
    prominences and stray bright pixels beyond the limb.

    Args:
        image: Two-dimensional full-disk image.
        fill_holes: Fill interior holes before measuring, so that very dark
            filaments or sunspots do not eat into the measured disk area.

    Returns:
        The fitted :class:`SolarDisk`.
    """
    if image.ndim != 2:
        raise ValueError(f"expected a 2-D image, got shape {image.shape}")

    work = np.nan_to_num(image.astype(np.float32), nan=0.0)
    threshold = _otsu_threshold(work)
    binary = work > threshold

    if not binary.any():
        # Degenerate image: fall back to the geometric centre.
        half = min(image.shape) / 2.0
        return SolarDisk(image.shape[0] / 2.0, image.shape[1] / 2.0, half)

    labels, count = ndi.label(binary)
    if count > 1:
        sizes = ndi.sum_labels(binary, labels, index=np.arange(1, count + 1))
        binary = labels == (int(np.argmax(sizes)) + 1)
    if fill_holes:
        binary = ndi.binary_fill_holes(binary)

    centre_y, centre_x = ndi.center_of_mass(binary)
    radius = float(np.sqrt(binary.sum() / np.pi))
    return SolarDisk(float(centre_y), float(centre_x), radius)


def refine_disk(image: np.ndarray, disk: SolarDisk, n_rays: int = 360) -> SolarDisk:
    """Refine a disk estimate by fitting a circle to limb crossings.

    We march outwards along ``n_rays`` evenly spaced rays and take the limb to
    be the point of steepest intensity fall-off, located to sub-pixel accuracy
    by fitting a parabola to the gradient around its peak.  The steepest-gradient
    definition is used in preference to a half-intensity crossing because limb
    darkening already dims the disk substantially before the true edge, which
    biases any fixed-level crossing inwards by a couple of per cent.

    A circle is then fitted to the crossings by least squares, with one round of
    outlier rejection so that prominences and cloud edges cannot drag the fit.
    """
    work = np.nan_to_num(image.astype(np.float32), nan=0.0)
    # Light smoothing suppresses noise spikes that would otherwise masquerade
    # as the steepest gradient.
    work = ndi.gaussian_filter(work, 1.0, mode="nearest")

    angles = np.linspace(0.0, 2.0 * np.pi, n_rays, endpoint=False)
    step = 0.25
    steps = np.arange(0.80 * disk.radius, 1.20 * disk.radius, step)
    if steps.size < 8:
        return disk

    sin_a, cos_a = np.sin(angles), np.cos(angles)
    ys = disk.centre_y + steps[None, :] * sin_a[:, None]
    xs = disk.centre_x + steps[None, :] * cos_a[:, None]
    profiles = ndi.map_coordinates(work, [ys.ravel(), xs.ravel()], order=1, mode="constant")
    profiles = profiles.reshape(n_rays, steps.size)

    # Rays that leave the frame are unusable.
    inside = (
        (ys >= 0) & (ys < work.shape[0]) & (xs >= 0) & (xs < work.shape[1])
    ).all(axis=1)

    gradient = -np.gradient(profiles, axis=1)  # positive where intensity falls
    peak = np.argmax(gradient, axis=1)

    points: list[tuple[float, float]] = []
    for ray in range(n_rays):
        if not inside[ray]:
            continue
        index = int(peak[ray])
        if index <= 0 or index >= steps.size - 1:
            continue
        left, centre, right = gradient[ray, index - 1 : index + 2]
        if centre <= 0:
            continue
        # Parabolic interpolation of the gradient peak for sub-pixel accuracy.
        denominator = left - 2.0 * centre + right
        shift = 0.0 if abs(denominator) < 1e-9 else 0.5 * (left - right) / denominator
        shift = float(np.clip(shift, -1.0, 1.0))
        radius_here = steps[index] + shift * step
        points.append(
            (
                disk.centre_y + radius_here * sin_a[ray],
                disk.centre_x + radius_here * cos_a[ray],
            )
        )

    if len(points) < 20:
        return disk

    pts = np.asarray(points, dtype=np.float64)
    fitted = _fit_circle(pts)
    if fitted is None:
        return disk

    # One robust pass: drop crossings far from the first fit and refit.
    residual = np.abs(
        np.hypot(pts[:, 0] - fitted[0], pts[:, 1] - fitted[1]) - fitted[2]
    )
    keep = residual <= max(2.0, 2.5 * float(np.median(residual)))
    if keep.sum() >= 20:
        refit = _fit_circle(pts[keep])
        if refit is not None:
            fitted = refit

    centre_y, centre_x, radius = fitted
    # Reject wild fits, which can happen on heavily clouded frames.
    if abs(radius - disk.radius) > 0.15 * disk.radius:
        return disk
    return SolarDisk(float(centre_y), float(centre_x), float(radius))


def _fit_circle(points: np.ndarray) -> tuple[float, float, float] | None:
    """Algebraic least-squares circle fit to (y, x) points."""
    y, x = points[:, 0], points[:, 1]
    design = np.stack([2.0 * x, 2.0 * y, np.ones_like(x)], axis=1)
    target = x**2 + y**2
    try:
        solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    except np.linalg.LinAlgError:
        return None
    cx, cy, offset = solution
    radius_sq = offset + cx**2 + cy**2
    if not np.isfinite(radius_sq) or radius_sq <= 0:
        return None
    return float(cy), float(cx), float(np.sqrt(radius_sq))


def detect_disk(image: np.ndarray) -> SolarDisk:
    """Locate the solar disk and refine it. This is the usual entry point."""
    return refine_disk(image, find_disk(image))
