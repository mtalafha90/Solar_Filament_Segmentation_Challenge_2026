"""Tests for disk detection and photometric flattening."""

import numpy as np
import pytest

from filaseg.data.synthetic import generate_observation
from filaseg.preprocessing.disk import SolarDisk, detect_disk, find_disk
from filaseg.preprocessing.photometry import (
    normalise,
    preprocess,
    radial_limb_profile,
    remove_limb_darkening,
)


def test_find_disk_recovers_geometry(observation):
    disk = find_disk(observation.image)
    assert disk.radius == pytest.approx(observation.radius, rel=0.05)
    assert disk.centre_y == pytest.approx(observation.centre_y, abs=3.0)
    assert disk.centre_x == pytest.approx(observation.centre_x, abs=3.0)


def test_refined_disk_is_sub_pixel_accurate(observation):
    disk = detect_disk(observation.image)
    assert disk.radius == pytest.approx(observation.radius, abs=1.0)
    assert disk.centre_y == pytest.approx(observation.centre_y, abs=0.5)
    assert disk.centre_x == pytest.approx(observation.centre_x, abs=0.5)


def test_disk_mask_and_radial_map():
    disk = SolarDisk(50.0, 50.0, 25.0)
    radial = disk.radial_map((100, 100))
    assert radial[50, 50] == pytest.approx(0.0)
    assert radial[50, 75] == pytest.approx(1.0, abs=0.02)
    mask = disk.mask((100, 100))
    # Area of the mask should match pi r^2 to within a pixel or two of rounding.
    assert mask.sum() == pytest.approx(np.pi * 25.0**2, rel=0.02)


def test_limb_profile_decreases_outwards(observation):
    disk = detect_disk(observation.image)
    _, profile = radial_limb_profile(observation.image, disk)
    # Limb darkening means the disk is brightest at the centre.
    assert profile[:20].mean() > profile[-40:-20].mean()


def test_limb_darkening_removal_flattens_the_disk(observation):
    disk = detect_disk(observation.image)
    flat = remove_limb_darkening(observation.image, disk)
    radial = disk.radial_map(observation.image.shape)

    inner = (radial < 0.3)
    outer = (radial > 0.6) & (radial < 0.9)
    before = np.median(observation.image[inner]) / np.median(observation.image[outer])
    after = np.median(flat[inner]) / np.median(flat[outer])
    # The ratio should be much closer to unity after flattening.
    assert abs(after - 1.0) < abs(before - 1.0)
    assert after == pytest.approx(1.0, abs=0.05)


def test_preprocess_gives_uniform_filament_contrast(observation):
    processed, valid, disk = preprocess(observation.image)
    truth = observation.semantic_mask & valid
    radial = disk.radial_map(processed.shape)

    contrasts = []
    for low, high in [(0.0, 0.4), (0.4, 0.7), (0.7, 0.95)]:
        band = valid & (radial >= low) & (radial < high)
        filament = band & truth
        quiet = band & ~truth
        if filament.sum() > 50 and quiet.sum() > 50:
            contrasts.append(processed[filament].mean() - processed[quiet].mean())

    assert len(contrasts) >= 2
    # Filaments must be brighter than quiet Sun everywhere (the image is inverted)
    assert all(c > 0.1 for c in contrasts)
    # and by a similar amount regardless of position on the disk.
    assert max(contrasts) / min(contrasts) < 1.5


def test_preprocess_zeroes_off_disk(observation):
    processed, valid, _ = preprocess(observation.image)
    assert processed[~valid].max() == pytest.approx(0.0)
    assert valid.sum() > 0


def test_normalise_is_robust_to_outliers():
    values = np.concatenate([np.random.rand(1000), [1e6]])
    image = values.reshape(-1, 1).astype(np.float32)
    valid = np.ones_like(image, dtype=bool)
    out = normalise(image, valid)
    assert out.min() >= 0.0 and out.max() <= 1.0
    # A single extreme outlier must not compress everything else to zero.
    assert out[:1000].std() > 0.2


def test_preprocess_handles_a_blank_image():
    blank = np.zeros((64, 64), dtype=np.float32)
    processed, valid, disk = preprocess(blank)
    assert processed.shape == blank.shape
    assert np.isfinite(processed).all()
    assert disk.radius > 0


def test_preprocess_handles_nan():
    observation = generate_observation(size=128, n_filaments=2, seed=3)
    image = observation.image.copy()
    image[10:20, 10:20] = np.nan
    processed, _, _ = preprocess(image)
    assert np.isfinite(processed).all()


def test_smooth_background_matches_the_direct_computation():
    """Wide Gaussians are decimated for speed; the result must not shift."""
    from scipy import ndimage as ndi

    from filaseg.preprocessing.photometry import smooth_background

    rng = np.random.default_rng(0)
    values = rng.random((256, 256)).astype(np.float32)
    weights = np.zeros((256, 256), dtype=np.float32)
    weights[20:240, 20:240] = 1.0  # a validity mask with real edges

    for sigma in (16.0, 32.0, 48.0):
        exact = ndi.gaussian_filter(values * weights, sigma, mode="nearest") / np.maximum(
            ndi.gaussian_filter(weights, sigma, mode="nearest"), 1e-6
        )
        fast = smooth_background(values * weights, weights, sigma)
        inside = weights > 0
        relative = np.abs(fast - exact)[inside] / np.maximum(np.abs(exact)[inside], 1e-6)
        # Comfortably below the quantisation of an 8-bit source (1/255).
        assert np.median(relative) < 1e-2
        assert np.percentile(relative, 99) < 5e-2


def test_smooth_background_is_exact_for_narrow_kernels():
    from scipy import ndimage as ndi

    from filaseg.preprocessing.photometry import smooth_background

    values = np.random.default_rng(1).random((64, 64)).astype(np.float32)
    weights = np.ones((64, 64), dtype=np.float32)
    exact = ndi.gaussian_filter(values, 4.0, mode="nearest")
    assert np.allclose(smooth_background(values, weights, 4.0), exact, atol=1e-5)


def test_smooth_background_ignores_pixels_outside_the_mask():
    """Normalised convolution: zero-weight pixels must not pull the estimate down."""
    from filaseg.preprocessing.photometry import smooth_background

    values = np.zeros((128, 128), dtype=np.float32)
    weights = np.zeros((128, 128), dtype=np.float32)
    values[:64] = 5.0
    weights[:64] = 1.0  # only the top half is valid

    background = smooth_background(values, weights, 12.0)
    # Well inside the valid region the estimate must be the valid value, not an
    # average that has been dragged towards the empty half.
    assert background[20, 64] == pytest.approx(5.0, rel=0.02)


def test_ridge_response_is_exact_at_the_shipped_scales():
    """The shipped scales must not be decimated: it moves the detections."""
    from filaseg.classical import ClassicalConfig, ridge_response

    image = np.zeros((128, 128), dtype=np.float32)
    image[62:68, 10:118] = 1.0
    scales = ClassicalConfig().scales
    default = ridge_response(image, scales)
    exact = ridge_response(image, scales, max_direct_sigma=1e9)
    assert np.array_equal(default, exact)
