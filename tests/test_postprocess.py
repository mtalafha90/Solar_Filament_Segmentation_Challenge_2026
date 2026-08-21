"""Tests for instance extraction and the classical detector."""

import numpy as np
import pytest
from scipy import ndimage as ndi

from filaseg.classical import ClassicalConfig, detect, hysteresis, ridge_response
from filaseg.metrics import evaluate
from filaseg.postprocess.instances import (
    InstanceConfig,
    extract_instances,
    fill_holes,
    merge_collinear,
    reject_compact,
    remove_small,
    shape_descriptors,
)
from filaseg.preprocessing.photometry import preprocess


def _label(mask):
    labels, _ = ndi.label(mask, structure=np.ones((3, 3), dtype=int))
    return labels


def test_remove_small_drops_tiny_components():
    mask = np.zeros((64, 64), dtype=bool)
    mask[10:30, 10:30] = True  # 400 pixels
    mask[50, 50] = True  # 1 pixel
    labels = remove_small(_label(mask), min_area=10)
    assert labels.max() == 1


def test_fill_holes_fills_small_voids_only():
    mask = np.ones((64, 64), dtype=bool)
    mask[10:12, 10:12] = False  # 4-pixel hole
    mask[30:50, 30:50] = False  # 400-pixel hole
    filled = fill_holes(mask, max_area=16)
    assert filled[10, 10]
    assert not filled[40, 40]


def test_merge_collinear_joins_an_interrupted_filament():
    mask = np.zeros((200, 200), dtype=bool)
    mask[100:104, 20:80] = True
    mask[100:104, 92:150] = True  # same line, 12-pixel gap
    labels = _label(mask)
    assert labels.max() == 2
    assert merge_collinear(labels, max_gap=18, max_angle=45).max() == 1


def test_merge_collinear_keeps_parallel_filaments_apart():
    mask = np.zeros((200, 200), dtype=bool)
    mask[150:154, 20:90] = True
    mask[160:164, 20:90] = True  # parallel, only 6 pixels away
    assert merge_collinear(_label(mask), max_gap=18, max_angle=45).max() == 2


def test_merge_collinear_respects_the_gap_limit():
    mask = np.zeros((200, 200), dtype=bool)
    mask[100:104, 20:80] = True
    mask[100:104, 140:190] = True  # same line, but a 60-pixel gap
    assert merge_collinear(_label(mask), max_gap=18, max_angle=45).max() == 2


def test_shape_descriptors_separate_bars_from_discs():
    bar = np.zeros((64, 64), dtype=bool)
    bar[30:34, 5:60] = True
    disc = np.zeros((64, 64), dtype=bool)
    yy, xx = np.ogrid[:64, :64]
    disc[(yy - 32) ** 2 + (xx - 32) ** 2 <= 64] = True

    assert shape_descriptors(bar)["axis_ratio"] > 5.0
    assert shape_descriptors(disc)["axis_ratio"] < 1.5
    assert shape_descriptors(disc)["roundness"] > shape_descriptors(bar)["roundness"]


def test_reject_compact_removes_sunspots_and_keeps_filaments():
    mask = np.zeros((128, 128), dtype=bool)
    mask[60:64, 10:100] = True  # filament
    yy, xx = np.ogrid[:128, :128]
    mask[(yy - 20) ** 2 + (xx - 20) ** 2 <= 49] = True  # round sunspot

    labels = reject_compact(_label(mask), max_area=900, min_axis_ratio=1.7)
    assert labels.max() == 1
    kept = labels == 1
    assert kept[62, 50]  # the filament survived
    assert not kept[20, 20]  # the sunspot did not


def test_reject_compact_keeps_large_round_regions():
    mask = np.zeros((256, 256), dtype=bool)
    yy, xx = np.ogrid[:256, :256]
    mask[(yy - 128) ** 2 + (xx - 128) ** 2 <= 40**2] = True  # ~5000 pixels
    assert reject_compact(_label(mask), max_area=900, min_axis_ratio=1.7).max() == 1


def test_extract_instances_runs_the_whole_chain():
    probability = np.zeros((200, 200), dtype=np.float32)
    probability[100:104, 20:80] = 1.0
    probability[100:104, 92:150] = 1.0
    yy, xx = np.ogrid[:200, :200]
    probability[(yy - 40) ** 2 + (xx - 40) ** 2 <= 64] = 1.0

    # Sunspot rejection is off by default, so the round blob survives.
    kept = extract_instances(probability, config=InstanceConfig(min_area=30))
    assert kept.max() == 2  # fragments merged; sunspot still present
    assert kept.dtype == np.int32

    # Switching it on, as the classical detector does, removes the sunspot.
    labels = extract_instances(
        probability, config=InstanceConfig(min_area=30, reject_round=True)
    )
    assert labels.max() == 1


def test_extract_instances_on_an_empty_map():
    labels = extract_instances(np.zeros((32, 32), dtype=np.float32))
    assert labels.max() == 0


def test_extract_instances_respects_the_valid_mask():
    probability = np.ones((64, 64), dtype=np.float32)
    valid = np.zeros((64, 64), dtype=bool)
    valid[:32, :32] = True
    labels = extract_instances(probability, valid, InstanceConfig(min_area=10))
    assert not (labels > 0)[40:, 40:].any()


def test_hysteresis_keeps_faint_material_attached_to_a_core():
    score = np.zeros((64, 64), dtype=np.float32)
    score[30:34, 10:40] = 0.9  # confident core
    score[31:33, 40:55] = 0.5  # faint barb, attached
    score[10:12, 10:20] = 0.5  # equally faint, but isolated

    kept = hysteresis(score, low=0.4, high=0.8)
    assert kept[32, 45]  # the attached barb survives
    assert not kept[11, 15]  # the isolated patch does not


def test_hysteresis_without_any_seed_returns_nothing():
    score = np.full((32, 32), 0.5, dtype=np.float32)
    assert not hysteresis(score, low=0.4, high=0.9).any()


def test_ridge_response_prefers_ridges_over_blobs():
    image = np.zeros((128, 128), dtype=np.float32)
    image[62:66, 10:60] = 1.0  # a bright ridge
    yy, xx = np.ogrid[:128, :128]
    image[(yy - 30) ** 2 + (xx - 90) ** 2 <= 16] = 1.0  # a bright blob of similar width

    response = ridge_response(image, (1.5, 2.5, 4.0))
    ridge_peak = response[62:66, 10:60].max()
    blob_peak = response[22:38, 82:98].max()
    assert ridge_peak > blob_peak


def test_classical_detector_finds_synthetic_filaments(observation):
    """The coverage prior has to match the data, as the documentation says.

    Synthetic frames are far denser than real GONG observations -- a few per
    cent of the disk against MAGFiLO's 0.84% -- so the default, which is set for
    real data, under-segments them. This is the workflow the docs prescribe:
    measure the coverage, then set the prior from it.
    """
    _, valid, _ = preprocess(observation.image)
    truth = observation.instance_map
    coverage = (observation.semantic_mask & valid).sum() / valid.sum()

    labels = detect(observation.image, ClassicalConfig(expected_coverage=coverage / 3))
    scores = evaluate(labels, truth, valid)
    assert scores["iou"] > 0.4
    assert scores["precision"] > 0.6
    assert scores["hit_rate"] > 0.3


def test_default_coverage_prior_suits_real_gong_statistics():
    """MAGFiLO filaments cover ~0.84% of the disk; the default targets that."""
    from filaseg.classical import ClassicalConfig, choose_thresholds

    # A uniform score map makes the percentile land exactly where the prior asks.
    scores = np.linspace(0.0, 1.0, 100001)
    _, high = choose_thresholds(scores, ClassicalConfig())
    seeded = float((scores >= high).mean())
    assert 0.002 < seeded < 0.008
    # Hysteresis then grows several times that, landing near real coverage.
    low, _ = choose_thresholds(scores, ClassicalConfig())
    eligible = float((scores >= low).mean())
    assert eligible > seeded * 2


def test_classical_recall_improves_when_the_coverage_prior_matches():
    """The coverage prior is the detector's main knob, and it should behave."""
    from filaseg.data.synthetic import generate_observation

    observation = generate_observation(size=256, n_filaments=8, n_sunspots=2, seed=11)
    _, valid, _ = preprocess(observation.image)
    truth = observation.instance_map
    actual = (observation.semantic_mask & valid).sum() / valid.sum()

    low_prior = evaluate(
        detect(observation.image, ClassicalConfig(expected_coverage=0.004)), truth, valid
    )
    matched = evaluate(
        detect(observation.image, ClassicalConfig(expected_coverage=actual / 2)),
        truth,
        valid,
    )
    # Seeding far too little of the disk costs recall.
    assert matched["recall"] > low_prior["recall"]


def test_choose_thresholds_follows_the_coverage_prior():
    from filaseg.classical import choose_thresholds

    scores = np.linspace(0.0, 1.0, 10000)
    low_a, high_a = choose_thresholds(scores, ClassicalConfig(expected_coverage=0.01))
    low_b, high_b = choose_thresholds(scores, ClassicalConfig(expected_coverage=0.05))
    # A larger coverage prior means a lower seeding threshold.
    assert high_b < high_a
    assert low_b < low_a
    assert low_a <= high_a


def test_choose_thresholds_honours_explicit_overrides():
    from filaseg.classical import choose_thresholds

    scores = np.linspace(0.0, 1.0, 10001)
    low, high = choose_thresholds(
        scores, ClassicalConfig(high_percentile=90.0, low_percentile=80.0)
    )
    assert high == pytest.approx(0.9, abs=0.01)
    assert low == pytest.approx(0.8, abs=0.01)


def test_classical_detector_on_a_blank_disk():
    image = np.zeros((128, 128), dtype=np.float32)
    yy, xx = np.ogrid[:128, :128]
    image[(yy - 64) ** 2 + (xx - 64) ** 2 <= 50**2] = 1.0
    labels = detect(image)
    # A featureless disk should produce few or no detections.
    assert labels.max() <= 2


def test_classical_config_is_overridable(observation):
    strict = ClassicalConfig(expected_coverage=0.002, growth_factor=2.0)
    loose = ClassicalConfig(expected_coverage=0.030, growth_factor=3.0)
    assert (detect(observation.image, strict) > 0).sum() < (
        detect(observation.image, loose) > 0
    ).sum()


def test_sunspot_rejection_is_off_by_default():
    """The default must not delete detections; the classical detector opts in.

    A trained network already ignores sunspots, so applying the shape filter to
    its output only removes genuine short filaments. The classical detector
    cannot tell the two apart, so it enables the filter explicitly.
    """
    from filaseg.classical import ClassicalConfig
    from filaseg.postprocess.instances import InstanceConfig

    assert InstanceConfig().reject_round is False
    assert ClassicalConfig().instance.reject_round is True


def test_default_post_processing_keeps_a_compact_filament():
    probability = np.zeros((128, 128), dtype=np.float32)
    # A short, stubby filament that a shape filter would wrongly discard.
    probability[60:70, 40:58] = 1.0
    labels = extract_instances(probability, config=InstanceConfig(min_area=30))
    assert labels.max() == 1


def test_filter_scales_follow_the_solar_radius():
    """A filament's width in pixels depends entirely on the plate scale."""
    from filaseg.classical import ClassicalConfig, scale_to_disk

    config = ClassicalConfig()
    at_reference = scale_to_disk(config, config.reference_radius)
    assert at_reference.scales == config.scales

    # A GONG frame's disk is about four times the reference radius.
    scaled = scale_to_disk(config, 4 * config.reference_radius)
    assert scaled.scales == pytest.approx(tuple(4 * s for s in config.scales))
    assert scaled.background_scale == pytest.approx(4 * config.background_scale)


def test_scale_to_disk_can_be_switched_off():
    from filaseg.classical import ClassicalConfig, scale_to_disk

    config = ClassicalConfig(scale_with_radius=False)
    assert scale_to_disk(config, 2000.0).scales == config.scales


def test_scale_to_disk_ignores_a_nonsensical_radius():
    from filaseg.classical import ClassicalConfig, scale_to_disk

    config = ClassicalConfig()
    for radius in (0.0, -5.0, float("nan")):
        assert scale_to_disk(config, radius).scales == config.scales


def test_detector_scales_itself_to_the_frame():
    """The same Sun at two resolutions must be detected about equally well."""
    from filaseg.classical import ClassicalConfig, detect
    from filaseg.data.synthetic import generate_observation
    from filaseg.metrics import evaluate
    from filaseg.preprocessing.photometry import preprocess

    scores = {}
    for size, width_scale in ((256, 1.0), (512, 2.0)):
        observation = generate_observation(
            size=size, n_filaments=6, n_sunspots=3, seed=17, width_scale=width_scale
        )
        _, valid, _ = preprocess(observation.image)
        coverage = (observation.semantic_mask & valid).sum() / valid.sum()
        labels = detect(
            observation.image, ClassicalConfig(expected_coverage=coverage / 3)
        )
        scores[size] = evaluate(labels, observation.instance_map, valid)["iou"]

    assert scores[256] > 0.3 and scores[512] > 0.3
    # Neither resolution should be dramatically worse than the other.
    assert min(scores.values()) > 0.6 * max(scores.values())


def test_instance_lengths_follow_the_solar_radius():
    """Every length describes the Sun, not the sensor, so all scale together."""
    from filaseg.postprocess.instances import InstanceConfig, scale_to_disk

    config = InstanceConfig()
    assert scale_to_disk(config, config.reference_radius).merge_gap == config.merge_gap

    scaled = scale_to_disk(config, 4 * config.reference_radius)
    assert scaled.merge_gap == pytest.approx(4 * config.merge_gap)
    # Areas scale with the square.
    assert scaled.fill_hole_area == pytest.approx(16 * config.fill_hole_area, rel=0.01)
    assert scaled.max_roundness_area == pytest.approx(
        16 * config.max_roundness_area, rel=0.01
    )
    # min_area is left alone: it already scales through min_area_fraction.
    assert scaled.min_area == config.min_area


def test_instance_scaling_can_be_switched_off():
    from filaseg.postprocess.instances import InstanceConfig, scale_to_disk

    config = InstanceConfig(scale_with_radius=False)
    assert scale_to_disk(config, 2000.0).merge_gap == config.merge_gap


def test_radius_is_recovered_from_the_disk_mask():
    from filaseg.postprocess.instances import radius_from_mask

    yy, xx = np.ogrid[:512, :512]
    valid = ((yy - 256) ** 2 + (xx - 256) ** 2) <= 200**2
    assert radius_from_mask(valid) == pytest.approx(200.0, rel=0.01)
    assert radius_from_mask(np.zeros((16, 16), dtype=bool)) == 0.0


def test_a_wide_gap_is_bridged_only_at_the_matching_resolution():
    """The same filament, imaged at two scales, must be rejoined at both."""
    from filaseg.postprocess.instances import InstanceConfig, extract_instances

    for factor in (1, 4):
        size = 512 * factor
        radius = 225 * factor
        yy, xx = np.ogrid[:size, :size]
        valid = ((yy - size // 2) ** 2 + (xx - size // 2) ** 2) <= radius**2

        probability = np.zeros((size, size), dtype=np.float32)
        row = size // 2
        thickness = 4 * factor
        # Two collinear fragments separated by a gap of 12 reference pixels.
        probability[row : row + thickness, 40 * factor : 200 * factor] = 1.0
        probability[row : row + thickness, 212 * factor : 380 * factor] = 1.0

        labels = extract_instances(probability, valid, InstanceConfig())
        assert labels.max() == 1, f"not rejoined at {size}px"


def test_confidence_filtering_removes_marginal_instances():
    """Area cannot separate these: the spurious blobs are as large as the real one."""
    from filaseg.postprocess.instances import InstanceConfig, extract_instances

    yy, xx = np.ogrid[:1024, :1024]
    valid = ((yy - 512) ** 2 + (xx - 512) ** 2) <= 450**2
    probability = np.zeros((1024, 1024), dtype=np.float32)
    probability[500:508, 200:700] = 0.95   # confident filament
    probability[300:308, 200:700] = 0.55   # marginal, same size
    probability[700:706, 300:500] = 0.52   # marginal

    assert extract_instances(probability, valid,
                             InstanceConfig(threshold=0.5)).max() == 3
    assert extract_instances(
        probability, valid, InstanceConfig(threshold=0.5, min_confidence=0.6)
    ).max() == 1


def test_peak_confidence_keeps_a_faint_filament_with_a_confident_core():
    from filaseg.postprocess.instances import InstanceConfig, extract_instances

    yy, xx = np.ogrid[:1024, :1024]
    valid = ((yy - 512) ** 2 + (xx - 512) ** 2) <= 450**2
    probability = np.zeros((1024, 1024), dtype=np.float32)
    probability[500:508, 200:700] = 0.55   # mostly faint
    probability[502:506, 400:460] = 0.98   # but with a sure core

    # A mean-only filter would discard it; the peak rule keeps it.
    assert extract_instances(
        probability, valid, InstanceConfig(threshold=0.5, min_confidence=0.7)
    ).max() == 0
    assert extract_instances(
        probability, valid,
        InstanceConfig(threshold=0.5, min_peak_confidence=0.9),
    ).max() == 1


def test_instance_confidence_reports_mean_and_peak():
    from filaseg.postprocess.instances import instance_confidence

    labels = np.zeros((64, 64), dtype=np.int32)
    labels[:8, :8] = 1
    probability = np.zeros((64, 64), dtype=np.float32)
    probability[:8, :8] = 0.5
    probability[0, 0] = 1.0

    mean_probability, peak = instance_confidence(labels, probability)[1]
    assert peak == pytest.approx(1.0)
    assert 0.5 < mean_probability < 0.52
