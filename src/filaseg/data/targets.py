"""Building the supervision targets a multi-task filament model needs.

Beyond the obvious binary mask we derive two auxiliary targets:

* a **spine (centreline) heatmap**, which tells the network where the
  topological core of each filament runs, and
* a **boundary map**, which concentrates loss on the outline where barbs live.

Both exist because a plain Dice loss on the binary mask is dominated by the
wide body of each filament.  Barbs are only a few pixels across, so getting
them wrong barely moves the Dice score, and models trained on Dice alone
reliably shave them off.  Supervising the centreline and the boundary directly
puts gradient where the fine structure actually is.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi


def skeletonise(mask: np.ndarray) -> np.ndarray:
    """Reduce a mask to a one-pixel-wide centreline."""
    from skimage.morphology import skeletonize

    if not mask.any():
        return np.zeros_like(mask, dtype=bool)
    return skeletonize(mask.astype(bool))


def spine_heatmap(
    mask: np.ndarray,
    spine_points: np.ndarray | None = None,
    sigma: float = 2.0,
) -> np.ndarray:
    """Build a soft centreline heatmap in ``[0, 1]``.

    When the dataset supplies an explicit spine we rasterise that, because it is
    the annotator's judgement of where the filament's axis runs.  Otherwise we
    fall back to the morphological skeleton of the mask.
    """
    canvas = np.zeros(mask.shape, dtype=np.float32)
    drawn = False

    if spine_points is not None and len(spine_points) >= 2:
        points = np.asarray(spine_points, dtype=np.float64)
        # Join consecutive spine vertices so the curve is continuous even when
        # the annotation is sparsely sampled.
        for start, end in zip(points[:-1], points[1:]):
            distance = float(np.hypot(*(end - start)))
            n_samples = max(2, int(distance) + 1)
            ys = np.linspace(start[0], end[0], n_samples)
            xs = np.linspace(start[1], end[1], n_samples)
            rows = np.clip(np.round(ys).astype(int), 0, mask.shape[0] - 1)
            cols = np.clip(np.round(xs).astype(int), 0, mask.shape[1] - 1)
            canvas[rows, cols] = 1.0
        drawn = canvas.any()

    if not drawn:
        canvas[skeletonise(mask)] = 1.0

    if sigma > 0 and canvas.any():
        canvas = ndi.gaussian_filter(canvas, sigma, mode="constant")
        peak = float(canvas.max())
        if peak > 0:
            canvas /= peak
    return canvas.astype(np.float32)


def boundary_map(mask: np.ndarray, width: int = 2) -> np.ndarray:
    """Mark a band of ``width`` pixels either side of every filament outline."""
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.float32)
    structure = ndi.generate_binary_structure(2, 2)
    dilated = ndi.binary_dilation(mask, structure, iterations=width)
    eroded = ndi.binary_erosion(mask, structure, iterations=width)
    return (dilated & ~eroded).astype(np.float32)


def distance_weight(
    mask: np.ndarray,
    valid: np.ndarray | None = None,
    base: float = 1.0,
    boundary_gain: float = 3.0,
    thin_gain: float = 2.0,
    width: int = 2,
) -> np.ndarray:
    """Per-pixel loss weights that emphasise outlines and thin extremities.

    Three contributions are summed:

    * ``base`` everywhere valid,
    * ``boundary_gain`` on the band around each filament outline,
    * ``thin_gain`` scaled by how thin the local filament is, measured by the
      distance transform inside the mask.  A barb two pixels wide gets the full
      bonus; the fat body of a filament gets almost none.
    """
    weights = np.full(mask.shape, base, dtype=np.float32)
    weights += boundary_gain * boundary_map(mask, width=width)

    if mask.any():
        inside = ndi.distance_transform_edt(mask).astype(np.float32)
        # Normalise by the thickest point so the bonus is scale free, then
        # invert so thin regions score highest.
        thickest = float(inside.max())
        if thickest > 0:
            thinness = np.zeros_like(inside)
            thinness[mask] = 1.0 - inside[mask] / thickest
            weights += thin_gain * thinness

    if valid is not None:
        weights *= valid.astype(np.float32)
    return weights
