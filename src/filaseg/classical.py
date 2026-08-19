"""A training-free filament detector.

This is a complete classical baseline: no network, no weights, no training
data.  It is here for three reasons.  It gives an immediate reference score to
beat; it runs on any H-alpha frame from any instrument without retraining; and
it is a safe fallback for observations that look nothing like the training set.

The method is built around two observations about what a filament actually is.

*A filament is a dark ridge, not merely a dark pixel.*  Sunspots are darker
than most filaments, so any detector keyed on darkness alone finds sunspots
first.  A multi-scale ridge filter responds to elongated dark structures and
largely ignores round ones, which separates the two by shape rather than by
contrast.

*A filament is bright in the middle and faint at the edges.*  Its barbs are
close to the noise floor, so a single threshold either misses them or floods
the image.  Hysteresis thresholding solves this properly: a high threshold
seeds confident filament cores, and those seeds then grow outwards through
anything above a much lower threshold.  Faint barbs are recovered because they
are connected to a confident core, while equally faint noise elsewhere is not.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
from scipy import ndimage as ndi

from .postprocess.instances import InstanceConfig, extract_instances
from .preprocessing.disk import SolarDisk
from .preprocessing.photometry import preprocess, smooth_background


@dataclass
class ClassicalConfig:
    """Settings for :func:`detect`."""

    scales: tuple[float, ...] = (1.5, 2.5, 4.0, 6.0, 9.0)
    """Ridge filter widths in pixels, covering thin barbs to fat filament bodies.

    Quoted for a disk of ``reference_radius``. Unless ``scale_with_radius`` is
    turned off these are rescaled to the disk actually measured, because a
    filament's width in pixels depends entirely on the plate scale: the same
    structure spans four times as many pixels on a 2048-pixel GONG frame as on a
    512-pixel thumbnail, and a filter tuned for one is blind to the other.
    """
    scale_with_radius: bool = True
    """Rescale ``scales`` and ``background_scale`` to the measured solar radius."""
    reference_radius: float = 225.0
    """Solar radius, in pixels, that ``scales`` and ``background_scale`` assume."""
    ridge_weight: float = 0.3
    """Blend between the ridge response (1.0) and the plain intensity deficit (0.0).

    Tuned on synthetic data, where the intensity deficit carries most of the
    signal and the shape filter in
    :func:`~filaseg.postprocess.instances.reject_compact` does most of the
    sunspot rejection.  Re-tune on MAGFiLO with ``scripts/tune_classical.py``
    before drawing conclusions -- real sunspot groups are far more irregular
    than the synthetic ones, so the ridge term is expected to matter more.
    """
    expected_coverage: float = 0.004
    """Fraction of the solar disk expected to be covered by filament *cores*.

    This is the detector's one real assumption, and it is stated in physical
    terms rather than buried in a percentile: the seeding threshold is placed so
    that this fraction of the disk is marked as confident filament, and
    hysteresis then grows outwards from those seeds to pick up the faint barbs.
    Final coverage is larger than this number -- see ``growth_factor``.

    It genuinely matters, because filament coverage varies by an order of
    magnitude over the solar cycle, and seeding the wrong fraction caps either
    recall or precision no matter how good the score map is.

    The default is set for real GONG data: filaments cover about 0.84% of the
    disk across MAGFiLO, and cores are a fraction of that again. Synthetic
    frames from :mod:`filaseg.data.synthetic` are far denser and want a value
    several times higher. Measure it rather than guessing --
    ``scripts/inspect_data.py`` reports the observed coverage and
    ``scripts/tune_classical.py`` searches around it.
    """
    growth_factor: float = 4.0
    """How far hysteresis may grow beyond the seeds, as a multiple of coverage.

    The growing threshold is placed so that ``expected_coverage *
    growth_factor`` of the disk is *eligible* to join a filament.  Only material
    connected to a seed is actually kept, so this bounds the growth rather than
    setting it.  Raising it recovers fainter barbs at some cost in precision.
    """
    high_percentile: float | None = None
    """Explicit seeding percentile, overriding ``expected_coverage`` when set."""
    low_percentile: float | None = None
    """Explicit growing percentile, overriding the derived one when set."""
    background_scale: float = 48.0
    """Length scale of the local background estimate, in pixels."""
    instance: InstanceConfig = field(
        default_factory=lambda: InstanceConfig(reject_round=True)
    )
    """Post-processing settings.

    Shape-based sunspot rejection is switched on here, unlike the global
    default: this detector keys on darkness and elongation, so sunspots do get
    through and must be removed geometrically.  A trained network does not need
    it -- see :class:`~filaseg.postprocess.instances.InstanceConfig`.
    """


def _hessian_ridge(image: np.ndarray, sigma: float) -> np.ndarray:
    """Ridge strength at one scale, from the Hessian eigenvalues.

    A ridge has one large negative eigenvalue across its width and one near zero
    along its length; a blob has two large negative ones. Scoring the difference
    therefore rewards elongation and suppresses round features such as sunspots.

    Responses are normalised by ``sigma**2`` so that scales are comparable, which
    is what lets one threshold serve thin barbs and fat filament bodies alike.
    """
    dyy = ndi.gaussian_filter(image, sigma, order=(2, 0), mode="nearest") * sigma**2
    dxx = ndi.gaussian_filter(image, sigma, order=(0, 2), mode="nearest") * sigma**2
    dxy = ndi.gaussian_filter(image, sigma, order=(1, 1), mode="nearest") * sigma**2

    trace = dyy + dxx
    difference = dyy - dxx
    root = np.sqrt(np.maximum(difference**2 + 4.0 * dxy**2, 0.0))
    lambda1 = 0.5 * (trace + root)  # larger eigenvalue
    lambda2 = 0.5 * (trace - root)  # smaller (most negative) eigenvalue

    # A bright ridge: lambda2 strongly negative, lambda1 near zero.
    response = np.maximum(-lambda2, 0.0) - np.maximum(np.abs(lambda1), 0.0)
    return np.maximum(response, 0.0)


def ridge_response(
    image: np.ndarray,
    scales: tuple[float, ...],
    max_direct_sigma: float = 12.0,
) -> np.ndarray:
    """Multi-scale ridge strength for bright elongated structures.

    The image is expected to have filaments *bright*, which is what the
    preprocessing chain produces.

    Scales wider than ``max_direct_sigma`` are evaluated on a decimated grid and
    interpolated back. The sigma-squared normalisation makes the response
    scale-covariant, so halving the grid and halving sigma gives nearly the same
    answer.

    The default leaves the shipped scales exact, deliberately. Unlike the smooth
    background estimates -- where decimation is free, costing a relative error of
    3e-4 -- decimating the ridge filter measurably moves the detections: at a
    threshold of 3 pixels the predicted masks agreed with the exact ones at only
    0.97 IoU, and clDice fell by 0.011, for a 27% saving. Lower it if you need
    the speed and can afford that, but measure the effect on your data first.

    Args:
        image: Preprocessed image, filaments bright.
        scales: Ridge filter widths in pixels.
        max_direct_sigma: Widths above this are computed decimated. The default
            keeps every shipped scale exact.
    """
    best = np.zeros(image.shape, dtype=np.float32)
    for sigma in scales:
        factor = 1
        if sigma > max_direct_sigma:
            factor = max(1, int(sigma // max_direct_sigma))

        if factor > 1:
            from skimage.measure import block_reduce

            small = block_reduce(image, (factor, factor), np.mean)
            response = _hessian_ridge(small, sigma / factor)
            response = ndi.zoom(
                response,
                (image.shape[0] / response.shape[0], image.shape[1] / response.shape[1]),
                order=1,
                mode="nearest",
            )
            response = response[: image.shape[0], : image.shape[1]]
        else:
            response = _hessian_ridge(image, sigma)

        np.maximum(best, response.astype(np.float32), out=best)
    return best


def intensity_deficit(
    image: np.ndarray, valid: np.ndarray, scale: float = 48.0
) -> np.ndarray:
    """How far each pixel rises above its local background.

    Because the input has filaments bright, "rises above" is the filament
    signal.  The background is estimated by normalised convolution so that the
    off-disk region does not bleed in near the limb.
    """
    weights = valid.astype(np.float32)
    background = smooth_background(image * weights, weights, scale)
    return np.maximum(image - background, 0.0).astype(np.float32)


def hysteresis(
    score: np.ndarray, low: float, high: float, valid: np.ndarray | None = None
) -> np.ndarray:
    """Grow confident seeds through weaker but connected material.

    This is what recovers barbs.  A barb never exceeds the high threshold on its
    own, but it is attached to a filament core that does, so it is kept; an
    isolated noise excursion of the same amplitude is not.
    """
    weak = score >= low
    strong = score >= high
    if valid is not None:
        weak = weak & valid
        strong = strong & valid
    if not strong.any():
        return np.zeros(score.shape, dtype=bool)

    labels, count = ndi.label(weak, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return np.zeros(score.shape, dtype=bool)
    keep = np.zeros(count + 1, dtype=bool)
    keep[np.unique(labels[strong])] = True
    keep[0] = False
    return keep[labels]


def score_map(
    image: np.ndarray,
    valid: np.ndarray,
    config: ClassicalConfig | None = None,
) -> np.ndarray:
    """Combine the ridge response and the intensity deficit into one score.

    Both terms are rank-normalised over the disk before blending, because they
    have completely different units and dynamic ranges; comparing percentile
    positions is the only sane way to mix them.
    """
    config = config or ClassicalConfig()

    ridge = ridge_response(image, config.scales)
    deficit = intensity_deficit(image, valid, config.background_scale)

    ridge_n = _rank_normalise(ridge, valid)
    deficit_n = _rank_normalise(deficit, valid)
    blended = config.ridge_weight * ridge_n + (1.0 - config.ridge_weight) * deficit_n
    return (blended * valid).astype(np.float32)


def _rank_normalise(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Map values to their percentile rank within the valid region, in [0, 1]."""
    out = np.zeros(values.shape, dtype=np.float32)
    inside = values[valid]
    if inside.size == 0:
        return out
    if float(inside.max() - inside.min()) < 1e-12:
        # A constant map carries no information. Returning zeros keeps it from
        # contributing an arbitrary ordering to the blended score.
        return out
    order = np.argsort(inside, kind="stable")
    ranks = np.empty(inside.shape, dtype=np.float32)
    ranks[order] = np.linspace(0.0, 1.0, inside.size, dtype=np.float32)
    out[valid] = ranks
    return out


def scale_to_disk(config: ClassicalConfig, radius: float) -> ClassicalConfig:
    """Rescale the filter widths to the disk actually measured.

    Filament widths are a property of the Sun, not of the sensor, so every
    length in the detector has to follow the plate scale. Without this, the
    defaults silently stop working the moment the data is distributed at a
    different resolution from the one they were tuned at.

    Args:
        config: Settings whose lengths are quoted for ``reference_radius``.
        radius: Solar radius measured on this frame, in pixels.

    Returns:
        A copy with ``scales`` and ``background_scale`` rescaled, or the
        original when ``scale_with_radius`` is off.
    """
    if not config.scale_with_radius or config.reference_radius <= 0:
        return config
    factor = float(radius) / float(config.reference_radius)
    if not np.isfinite(factor) or factor <= 0 or abs(factor - 1.0) < 0.05:
        return config
    return replace(
        config,
        scales=tuple(float(s) * factor for s in config.scales),
        background_scale=float(config.background_scale) * factor,
    )


def choose_thresholds(
    inside: np.ndarray, config: ClassicalConfig | None = None
) -> tuple[float, float]:
    """Pick the hysteresis thresholds for one frame.

    The score map is rank-normalised, so a percentile *is* a score value and
    coverage translates directly into a threshold.  The seed threshold marks the
    ``expected_coverage`` highest-scoring fraction of the disk; the growing
    threshold admits ``expected_coverage * growth_factor``, of which only the
    part connected to a seed survives.

    Otsu's method was tried here and rejected.  It assumes two classes of
    comparable size, whereas filaments are a per cent or two of the disk, so it
    over-segments sparse frames badly -- measured on synthetic data, a frame
    with 1.2% true coverage was split at 21%.  A stated coverage prior is both
    more accurate and far easier to reason about.

    Args:
        inside: Score values for on-disk pixels only.
        config: Detector settings.

    Returns:
        ``(low, high)`` thresholds for :func:`hysteresis`.
    """
    config = config or ClassicalConfig()
    if inside.size == 0:
        return 0.0, 1.0

    coverage = float(np.clip(config.expected_coverage, 1e-4, 0.5))
    high_percentile = (
        config.high_percentile
        if config.high_percentile is not None
        else 100.0 * (1.0 - coverage)
    )
    low_percentile = (
        config.low_percentile
        if config.low_percentile is not None
        else 100.0 * (1.0 - min(coverage * config.growth_factor, 0.6))
    )

    high = float(np.percentile(inside, float(np.clip(high_percentile, 50.0, 99.99))))
    low = float(np.percentile(inside, float(np.clip(low_percentile, 1.0, high_percentile))))
    return float(min(low, high)), float(high)


def detect(
    image: np.ndarray,
    config: ClassicalConfig | None = None,
    disk: SolarDisk | None = None,
    return_score: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Detect filaments in a raw full-disk H-alpha image, without any training.

    Args:
        image: Raw full-disk image.
        config: Detector settings.
        disk: Pre-computed disk geometry; detected automatically when omitted.
        return_score: Also return the score map and the on-disk mask.

    Returns:
        An ``int32`` instance label map, or ``(labels, score, valid)`` when
        ``return_score`` is set.
    """
    config = config or ClassicalConfig()
    processed, valid, fitted = preprocess(image, disk=disk)
    config = scale_to_disk(config, fitted.radius)

    score = score_map(processed, valid, config)
    inside = score[valid]
    if inside.size == 0:
        empty = np.zeros(image.shape, dtype=np.int32)
        return (empty, score, valid) if return_score else empty

    low, high = choose_thresholds(inside, config)
    mask = hysteresis(score, low, high, valid)

    labels = extract_instances(mask.astype(np.float32), valid, config.instance)
    if return_score:
        return labels, score, valid
    return labels
