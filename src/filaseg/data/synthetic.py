"""A synthetic stand-in for GONG H-alpha observations.

The competition data (MAGFiLO) has to be downloaded from Kaggle.  This module
generates full-disk images that share the properties that actually matter for
algorithm design, so that the whole pipeline can be exercised, tested and
profiled before the real data is in place:

* a limb-darkened disk on a black sky, scaled to a nominal solar radius;
* smooth transmission gradients standing in for haze and flat-field residuals;
* photon and detector noise;
* dark, round sunspots, which are the main false-positive trap;
* bright plage and a chromospheric network texture;
* dark, elongated, curved filaments carrying fine barbs.

Filaments are drawn from a spine so we get an exact spine annotation for free,
matching the structure of the MAGFiLO annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi


@dataclass
class SyntheticFilament:
    """Ground truth for one generated filament."""

    spine: np.ndarray  # (N, 2) array of (y, x) points along the spine
    mask: np.ndarray  # boolean mask, full image size
    chirality: int  # 0 = unknown, 1 = left (sinistral), 2 = right (dextral)


@dataclass
class SyntheticObservation:
    """A generated full-disk image and everything known about it."""

    image: np.ndarray
    centre_y: float
    centre_x: float
    radius: float
    filaments: list[SyntheticFilament] = field(default_factory=list)

    @property
    def semantic_mask(self) -> np.ndarray:
        """Union of all filament masks."""
        out = np.zeros(self.image.shape, dtype=bool)
        for filament in self.filaments:
            out |= filament.mask
        return out

    @property
    def instance_map(self) -> np.ndarray:
        """Integer label map, 0 for background and 1..N for filaments."""
        out = np.zeros(self.image.shape, dtype=np.int32)
        for index, filament in enumerate(self.filaments, start=1):
            out[filament.mask] = index
        return out


def _limb_darkening(radius_map: np.ndarray, u: float = 0.75) -> np.ndarray:
    """Classical linear limb-darkening law, I(mu)/I(0) = 1 - u*(1 - mu)."""
    mu = np.sqrt(np.clip(1.0 - np.clip(radius_map, 0.0, 1.0) ** 2, 0.0, 1.0))
    return 1.0 - u * (1.0 - mu)


def _smooth_noise(shape: tuple[int, int], scale: float, rng: np.random.Generator) -> np.ndarray:
    """Zero-mean noise with a chosen correlation length, normalised to unit spread."""
    field_ = rng.standard_normal(shape).astype(np.float32)
    field_ = ndi.gaussian_filter(field_, scale, mode="wrap")
    spread = float(field_.std())
    return field_ / spread if spread > 1e-8 else field_


def _random_spine(
    start_y: float,
    start_x: float,
    length: float,
    rng: np.random.Generator,
    curvature: float = 0.05,
    step: float = 2.0,
) -> np.ndarray:
    """Trace a smoothly curving path, which becomes a filament's spine."""
    n_steps = max(4, int(length / step))
    angle = rng.uniform(0.0, 2.0 * np.pi)
    # A slowly varying turn rate gives realistic, gently sinuous shapes.
    turn = ndi.gaussian_filter1d(
        rng.standard_normal(n_steps).astype(np.float32), max(1.0, n_steps / 8.0)
    )
    turn *= curvature / max(float(np.abs(turn).max()), 1e-6)

    points = np.empty((n_steps, 2), dtype=np.float32)
    y, x = start_y, start_x
    for index in range(n_steps):
        points[index] = (y, x)
        angle += float(turn[index])
        y += step * np.sin(angle)
        x += step * np.cos(angle)
    return points


def _stamp_path(
    canvas: np.ndarray,
    points: np.ndarray,
    widths: np.ndarray,
) -> None:
    """Paint a variable-width tube along ``points`` into ``canvas`` (in place).

    Each spine point contributes a small disc; taking the maximum across points
    yields a smooth tube.  Working on a local window per point keeps this fast
    even for a 2048-pixel canvas.
    """
    height, width = canvas.shape
    for (cy, cx), half in zip(points, widths):
        if half <= 0:
            continue
        reach = int(np.ceil(half)) + 1
        y0, y1 = max(0, int(cy) - reach), min(height, int(cy) + reach + 1)
        x0, x1 = max(0, int(cx) - reach), min(width, int(cx) + reach + 1)
        if y0 >= y1 or x0 >= x1:
            continue
        yy, xx = np.ogrid[y0:y1, x0:x1]
        dist = np.hypot(yy - cy, xx - cx)
        # A soft edge, so the tube has a realistic intensity roll-off.
        blob = np.clip(1.0 - (dist / max(half, 1e-3)) ** 2, 0.0, 1.0)
        np.maximum(canvas[y0:y1, x0:x1], blob, out=canvas[y0:y1, x0:x1])


def _make_filament(
    shape: tuple[int, int],
    disk_centre: tuple[float, float],
    disk_radius: float,
    rng: np.random.Generator,
    width_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int] | None:
    """Generate one filament: its soft intensity profile, mask and spine."""
    centre_y, centre_x = disk_centre
    # Place the spine start somewhere on the disk, avoiding the very limb.
    for _ in range(20):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        radius = disk_radius * np.sqrt(rng.uniform(0.0, 0.80))
        start_y = centre_y + radius * np.sin(angle)
        start_x = centre_x + radius * np.cos(angle)
        if 0 <= start_y < shape[0] and 0 <= start_x < shape[1]:
            break
    else:
        return None

    length = rng.uniform(0.06, 0.55) * disk_radius
    spine = _random_spine(start_y, start_x, length, rng, curvature=rng.uniform(0.02, 0.12))

    # Keep only the part of the spine that lands on the disk.
    on_disk = np.hypot(spine[:, 0] - centre_y, spine[:, 1] - centre_x) < 0.93 * disk_radius
    if on_disk.sum() < 6:
        return None
    spine = spine[on_disk]

    # Filaments are thicker in the body and taper at both ends.
    n_points = len(spine)
    along = np.linspace(0.0, 1.0, n_points)
    base_half_width = rng.uniform(2.5, 9.0) * width_scale
    taper = np.sin(np.pi * np.clip(along, 0.0, 1.0)) ** 0.35
    wobble = 1.0 + 0.25 * ndi.gaussian_filter1d(
        rng.standard_normal(n_points).astype(np.float32), max(1.0, n_points / 6.0)
    )
    widths = np.clip(base_half_width * taper * wobble, 0.8, None)

    canvas = np.zeros(shape, dtype=np.float32)
    _stamp_path(canvas, spine, widths)

    # Barbs: short, thin side-threads leaving the spine at a steep angle.
    # These are the fine-scale structures the challenge cares about most, and
    # the first thing a Dice-only model throws away.
    chirality = int(rng.integers(1, 3))
    barb_sign = 1.0 if chirality == 1 else -1.0
    n_barbs = int(rng.integers(0, 6))
    for _ in range(n_barbs):
        if n_points < 8:
            break
        anchor = int(rng.integers(2, n_points - 2))
        tangent = spine[min(anchor + 1, n_points - 1)] - spine[max(anchor - 1, 0)]
        norm = float(np.hypot(*tangent))
        if norm < 1e-6:
            continue
        tangent = tangent / norm
        # Rotate the tangent by roughly +/- 60 degrees to get the barb direction.
        theta = barb_sign * rng.uniform(np.pi / 4, np.pi / 2)
        rot = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ],
            dtype=np.float32,
        )
        direction = rot @ tangent
        barb_length = rng.uniform(6.0, 28.0) * width_scale
        n_barb_points = max(3, int(barb_length / 1.5))
        offsets = np.linspace(0.0, barb_length, n_barb_points)[:, None]
        barb_points = spine[anchor][None, :] + offsets * direction[None, :]
        barb_widths = np.linspace(
            max(1.6 * width_scale, base_half_width * 0.45), 0.9 * width_scale, n_barb_points
        )
        _stamp_path(canvas, barb_points, barb_widths)

    mask = canvas > 0.22
    if mask.sum() < 40:
        return None
    return canvas, mask, chirality


def generate_observation(
    size: int = 512,
    n_filaments: int = 8,
    n_sunspots: int = 4,
    solar_radius_fraction: float = 0.88,
    seed: int | None = None,
    noise_level: float = 0.012,
    filament_depth: tuple[float, float] = (0.30, 0.62),
    network_amplitude: float = 0.05,
    plage_amplitude: float = 0.07,
    width_scale: float = 1.0,
    limb_rim: float = 0.0,
) -> SyntheticObservation:
    """Generate one synthetic full-disk H-alpha observation with ground truth.

    Args:
        size: Side length of the square image in pixels.
        n_filaments: Target number of filaments (a few may be rejected as too
            small, so the final count can be lower).
        n_sunspots: Number of dark, round sunspots to add as distractors.
        solar_radius_fraction: Solar radius as a fraction of half the image
            side.  GONG frames sit at roughly 900/1024.
        seed: Seed for reproducibility.
        noise_level: Standard deviation of the additive detector noise.
        filament_depth: Range of fractional darkening a filament imposes. The
            default is generous; real H-alpha filaments are often far shallower,
            which is what makes them hard, so lower this to make a harder set.
        network_amplitude: Strength of the chromospheric network mottling, the
            main texture a detector has to see past.
        plage_amplitude: Strength of the bright plage patches.
        width_scale: Multiplies filament widths, for generating at a different
            resolution while keeping the apparent shape.
        limb_rim: Strength of a bright ring just inside the limb, standing in
            for the chromospheric emission and spicules that real H-alpha shows
            there. This is the single most destructive artefact for a detector
            that keys on local intensity: after limb-darkening is divided out,
            the rim survives as an enormous positive residual and dominates the
            score map unless the outer annulus is excluded.

    Returns:
        A :class:`SyntheticObservation`.
    """
    rng = np.random.default_rng(seed)
    shape = (size, size)
    centre_y = size / 2.0 + rng.uniform(-size * 0.01, size * 0.01)
    centre_x = size / 2.0 + rng.uniform(-size * 0.01, size * 0.01)
    radius = solar_radius_fraction * size / 2.0

    yy, xx = np.ogrid[:size, :size]
    radius_map = np.hypot(yy - centre_y, xx - centre_x) / radius
    disk = radius_map <= 1.0

    # Quiet Sun with limb darkening.
    image = _limb_darkening(radius_map, u=rng.uniform(0.6, 0.85)).astype(np.float32)

    # Chromospheric network: fine bright/dark mottling across the disk.
    image *= 1.0 + network_amplitude * _smooth_noise(shape, max(1.0, size / 256.0), rng)
    # Plage: larger bright patches.
    plage = _smooth_noise(shape, max(2.0, size / 40.0), rng)
    image *= 1.0 + plage_amplitude * np.clip(plage, 0.0, None)

    # Filaments, drawn dark against the disk.
    filaments: list[SyntheticFilament] = []
    attempts = 0
    while len(filaments) < n_filaments and attempts < n_filaments * 6:
        attempts += 1
        made = _make_filament(shape, (centre_y, centre_x), radius, rng, width_scale)
        if made is None:
            continue
        profile, mask, chirality = made
        if not (mask & disk).any():
            continue
        # Overlapping filaments would be ambiguous ground truth, so skip them.
        if any((mask & existing.mask).any() for existing in filaments):
            continue
        depth = rng.uniform(*filament_depth)
        image *= 1.0 - depth * profile
        spine_points = np.argwhere(
            ndi.binary_erosion(mask, np.ones((3, 3), dtype=bool), iterations=1)
        )
        filaments.append(
            SyntheticFilament(
                spine=spine_points.astype(np.float32),
                mask=mask & disk,
                chirality=chirality,
            )
        )

    # Sunspots: dark and round, the classic false positive for a filament
    # detector that keys only on darkness.
    for _ in range(n_sunspots):
        angle = rng.uniform(0.0, 2.0 * np.pi)
        rad = radius * np.sqrt(rng.uniform(0.0, 0.75))
        spot_y = centre_y + rad * np.sin(angle)
        spot_x = centre_x + rad * np.cos(angle)
        spot_radius = rng.uniform(size / 180.0, size / 60.0)
        dist = np.hypot(yy - spot_y, xx - spot_x)
        umbra = np.clip(1.0 - (dist / spot_radius) ** 2, 0.0, 1.0) ** 0.6
        image *= 1.0 - rng.uniform(0.35, 0.7) * umbra

    # A bright chromospheric rim just inside the limb, as real H-alpha shows.
    if limb_rim > 0:
        rim = np.exp(-(((radius_map - 0.985) / 0.012) ** 2))
        image *= 1.0 + limb_rim * rim

    # Ground-based effects: a smooth transmission gradient, then noise.
    image *= 1.0 + 0.10 * _smooth_noise(shape, size / 6.0, rng)
    image = image * disk
    image += noise_level * rng.standard_normal(shape).astype(np.float32)
    # Sky background is not perfectly black.
    image += 0.004 * rng.standard_normal(shape).astype(np.float32) * (~disk)
    image = np.clip(image, 0.0, None).astype(np.float32)

    return SyntheticObservation(
        image=image,
        centre_y=float(centre_y),
        centre_x=float(centre_x),
        radius=float(radius),
        filaments=filaments,
    )
