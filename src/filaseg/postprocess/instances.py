"""Turning a probability map into coherent filament instances.

Thresholding a segmentation map gives connected components, not filaments.  The
two differ in ways that matter for the challenge's instance metrics:

* **One filament often breaks into several components.**  A faint waist, a
  patch of poor seeing or a slightly conservative threshold splits a single
  long filament in two.  Scored as instances, that costs a hit and adds a false
  positive.  :func:`merge_collinear` repairs it by rejoining fragments whose
  ends face one another and whose spines line up -- the geometric signature of
  one interrupted structure rather than two neighbouring ones.

* **Not every dark blob is a filament.**  Sunspots are the main trap: dark,
  high contrast and completely round.  :func:`reject_compact` removes anything
  too round and too small to be a filament, using shape alone so that it
  transfers across instruments and observing conditions.

The order matters. Merge first, then filter on shape, because a fragment of a
long filament can look compact on its own.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass
class InstanceConfig:
    """Settings for :func:`extract_instances`."""

    threshold: float = 0.5
    min_area: int = 40
    """Discard components smaller than this many pixels. Scaled with resolution."""
    fill_hole_area: int = 64
    """Fill interior holes up to this area; larger voids are probably real."""
    merge_gap: float = 18.0
    """Largest gap in pixels across which two fragments may be rejoined."""
    merge_angle: float = 45.0
    """Largest misalignment in degrees permitted when rejoining fragments."""
    reject_round: bool = False
    """Remove compact, round blobs, which are almost always sunspots.

    Off by default, and that default is deliberate.  This filter is essential
    for the classical detector, which has no way to tell a sunspot from a
    filament and would otherwise report every one (measured: false discovery
    rate 0.60 without it, 0.14 with).  A trained network, however, has already
    learned to ignore sunspots -- it does not predict them at all -- so the
    filter finds nothing to remove and instead deletes genuine short, compact
    filaments.  Measured on synthetic validation frames, switching it on after
    FilaNet cost 0.145 IoU and dropped the hit rate from 0.96 to 0.73.

    Rule of thumb: enable it for any detector that cannot reject sunspots
    itself, and leave it off for one that can.
    """
    max_roundness_area: int = 900
    """Only blobs below this area are eligible for rejection as sunspots."""
    min_axis_ratio: float = 1.7
    """Minimum major/minor axis ratio for a small blob to be kept."""
    closing_radius: int = 0
    """Optional morphological closing applied before labelling."""


def _disk_structure(radius: int) -> np.ndarray:
    size = 2 * radius + 1
    yy, xx = np.ogrid[:size, :size]
    return (yy - radius) ** 2 + (xx - radius) ** 2 <= radius**2


def remove_small(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Drop components below ``min_area`` pixels and relabel consecutively."""
    if labels.max() == 0 or min_area <= 1:
        return labels
    counts = np.bincount(labels.ravel())
    keep = np.zeros(counts.shape[0], dtype=np.int32)
    next_label = 1
    for label in range(1, counts.shape[0]):
        if counts[label] >= min_area:
            keep[label] = next_label
            next_label += 1
    return keep[labels]


def fill_holes(mask: np.ndarray, max_area: int) -> np.ndarray:
    """Fill interior holes up to ``max_area`` pixels.

    Small holes are threshold noise inside a filament body.  Large ones may be
    genuine, so they are left alone.
    """
    if max_area <= 0 or not mask.any():
        return mask
    filled = ndi.binary_fill_holes(mask)
    holes = filled & ~mask
    if not holes.any():
        return mask
    hole_labels, count = ndi.label(holes)
    if count == 0:
        return mask
    sizes = np.bincount(hole_labels.ravel(), minlength=count + 1)
    small = np.zeros(count + 1, dtype=bool)
    small[1:] = sizes[1:] <= max_area
    return mask | small[hole_labels]


def _endpoints_and_directions(
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find a component's spine endpoints and the direction it leaves each one.

    Returns two arrays of shape ``(K, 2)``: the endpoint coordinates in
    ``(y, x)`` and, for each, the outward unit direction of the spine there.
    """
    from skimage.morphology import skeletonize

    skeleton = skeletonize(mask)
    if not skeleton.any():
        skeleton = mask

    coords = np.argwhere(skeleton)
    if len(coords) < 2:
        centre = np.argwhere(mask).mean(axis=0)
        return np.array([centre, centre]), np.zeros((2, 2))

    # A skeleton pixel with one neighbour is an endpoint.
    neighbours = ndi.convolve(
        skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant"
    )
    endpoint_mask = skeleton & (neighbours <= 2)
    endpoints = np.argwhere(endpoint_mask)

    if len(endpoints) < 2:
        # A closed loop or a blob: use the two extremes along the principal axis.
        centred = coords - coords.mean(axis=0)
        _, _, axes = np.linalg.svd(centred, full_matrices=False)
        projection = centred @ axes[0]
        endpoints = np.stack([coords[np.argmin(projection)], coords[np.argmax(projection)]])

    # Estimate each endpoint's direction from nearby skeleton pixels.
    directions = np.zeros((len(endpoints), 2), dtype=np.float64)
    for index, point in enumerate(endpoints):
        distance = np.hypot(coords[:, 0] - point[0], coords[:, 1] - point[1])
        near = coords[distance <= 9.0]
        if len(near) < 2:
            continue
        centred = near - point
        _, _, axes = np.linalg.svd(centred - centred.mean(axis=0), full_matrices=False)
        axis = axes[0]
        # Point the axis away from the component's body.
        if np.dot(centred.mean(axis=0), axis) > 0:
            axis = -axis
        norm = float(np.linalg.norm(axis))
        if norm > 1e-8:
            directions[index] = axis / norm
    return endpoints.astype(np.float64), directions


class _UnionFind:
    """Disjoint sets, used to group fragments that belong to one filament."""

    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, item: int) -> int:
        while self.parent[item] != item:
            self.parent[item] = self.parent[self.parent[item]]
            item = self.parent[item]
        return item

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[right_root] = left_root


def merge_collinear(
    labels: np.ndarray,
    max_gap: float = 18.0,
    max_angle: float = 45.0,
) -> np.ndarray:
    """Rejoin fragments that are two halves of one interrupted filament.

    Two components are merged when some endpoint of one lies within ``max_gap``
    of an endpoint of the other *and* both spines run roughly along the line
    joining them, within ``max_angle`` degrees.  Requiring the directions to
    agree is what stops two unrelated filaments that happen to pass close by
    from being welded together.

    Args:
        labels: Integer label map.
        max_gap: Largest gap to bridge, in pixels.
        max_angle: Largest angular disagreement, in degrees.

    Returns:
        A relabelled map in which merged fragments share a label.
    """
    n_labels = int(labels.max())
    if n_labels < 2 or max_gap <= 0:
        return labels

    endpoints: list[np.ndarray] = []
    directions: list[np.ndarray] = []
    objects = ndi.find_objects(labels)
    for index in range(1, n_labels + 1):
        window = objects[index - 1]
        if window is None:
            endpoints.append(np.zeros((0, 2)))
            directions.append(np.zeros((0, 2)))
            continue
        # Work in a padded window so skeletonisation is not affected by the crop.
        local = np.pad(labels[window] == index, 2)
        offset = np.array([window[0].start - 2, window[1].start - 2], dtype=np.float64)
        points, dirs = _endpoints_and_directions(local)
        endpoints.append(points + offset)
        directions.append(dirs)

    cosine_limit = float(np.cos(np.deg2rad(max_angle)))
    union = _UnionFind(n_labels + 1)

    for i in range(1, n_labels + 1):
        for j in range(i + 1, n_labels + 1):
            if union.find(i) == union.find(j):
                continue
            best = _best_bridge(
                endpoints[i - 1], directions[i - 1],
                endpoints[j - 1], directions[j - 1],
                max_gap, cosine_limit,
            )
            if best:
                union.union(i, j)

    # Relabel so the surviving groups are numbered 1..M.
    mapping = np.zeros(n_labels + 1, dtype=np.int32)
    next_label = 1
    for index in range(1, n_labels + 1):
        root = union.find(index)
        if mapping[root] == 0:
            mapping[root] = next_label
            next_label += 1
        mapping[index] = mapping[root]
    return mapping[labels]


def _best_bridge(
    points_a: np.ndarray,
    dirs_a: np.ndarray,
    points_b: np.ndarray,
    dirs_b: np.ndarray,
    max_gap: float,
    cosine_limit: float,
) -> bool:
    """Whether any endpoint pair justifies merging two components."""
    if len(points_a) == 0 or len(points_b) == 0:
        return False
    for index_a, point_a in enumerate(points_a):
        for index_b, point_b in enumerate(points_b):
            offset = point_b - point_a
            gap = float(np.hypot(*offset))
            if gap > max_gap or gap < 1e-6:
                continue
            bridge = offset / gap
            dir_a, dir_b = dirs_a[index_a], dirs_b[index_b]
            # Each spine must head towards the other across the gap.
            if np.linalg.norm(dir_a) > 1e-6 and float(np.dot(dir_a, bridge)) < cosine_limit:
                continue
            if np.linalg.norm(dir_b) > 1e-6 and float(np.dot(dir_b, -bridge)) < cosine_limit:
                continue
            return True
    return False


def shape_descriptors(mask: np.ndarray) -> dict[str, float]:
    """Area, axis ratio and roundness of one component.

    The axis ratio comes from the second moments of the pixel coordinates, so it
    is well defined even for curved filaments, where a bounding box would be
    misleading.
    """
    coords = np.argwhere(mask)
    area = float(len(coords))
    if area < 3:
        return {"area": area, "axis_ratio": 1.0, "roundness": 1.0}

    centred = coords - coords.mean(axis=0)
    covariance = np.cov(centred, rowvar=False)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    major = float(np.sqrt(max(eigenvalues[0], 1e-9)))
    minor = float(np.sqrt(max(eigenvalues[1], 1e-9)))
    axis_ratio = major / max(minor, 1e-6)

    # Roundness: 1 for a perfect disc, falling towards 0 for a long thin shape.
    equivalent_radius = np.sqrt(area / np.pi)
    roundness = float(np.clip(equivalent_radius / (2.0 * max(major, 1e-6)), 0.0, 1.0))
    return {"area": area, "axis_ratio": axis_ratio, "roundness": roundness}


def reject_compact(
    labels: np.ndarray,
    max_area: int = 900,
    min_axis_ratio: float = 1.7,
) -> np.ndarray:
    """Remove small, round components, which are overwhelmingly sunspots.

    Only components below ``max_area`` are eligible: a large round region is
    more likely to be a genuine filament complex than a sunspot, so leaving big
    objects alone keeps this from deleting real detections.
    """
    n_labels = int(labels.max())
    if n_labels == 0:
        return labels

    keep = np.zeros(n_labels + 1, dtype=np.int32)
    next_label = 1
    objects = ndi.find_objects(labels)
    for index in range(1, n_labels + 1):
        window = objects[index - 1]
        if window is None:
            continue
        descriptors = shape_descriptors(labels[window] == index)
        is_sunspot = (
            descriptors["area"] <= max_area
            and descriptors["axis_ratio"] < min_axis_ratio
        )
        if not is_sunspot:
            keep[index] = next_label
            next_label += 1
    return keep[labels]


def extract_instances(
    probability: np.ndarray,
    valid: np.ndarray | None = None,
    config: InstanceConfig | None = None,
) -> np.ndarray:
    """Convert a filament probability map into an instance label map.

    Args:
        probability: Per-pixel filament probability in ``[0, 1]``.
        valid: Optional on-disk mask; predictions outside it are discarded.
        config: Post-processing settings.

    Returns:
        An ``int32`` label map, 0 for background and 1..N for filaments.
    """
    config = config or InstanceConfig()
    mask = np.asarray(probability) >= config.threshold
    if valid is not None:
        mask = mask & np.asarray(valid, dtype=bool)
    if not mask.any():
        return np.zeros(mask.shape, dtype=np.int32)

    mask = fill_holes(mask, config.fill_hole_area)
    if config.closing_radius > 0:
        mask = ndi.binary_closing(mask, _disk_structure(config.closing_radius))

    # 8-connectivity, so diagonally touching barb pixels stay with their filament.
    labels, _ = ndi.label(mask, structure=np.ones((3, 3), dtype=int))
    labels = remove_small(labels, config.min_area)
    if labels.max() == 0:
        return labels.astype(np.int32)

    labels = merge_collinear(labels, config.merge_gap, config.merge_angle)
    if config.reject_round:
        labels = reject_compact(labels, config.max_roundness_area, config.min_axis_ratio)
    return labels.astype(np.int32)
