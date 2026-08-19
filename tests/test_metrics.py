"""Tests for the evaluation metrics."""

import numpy as np
import pytest

from filaseg.metrics import (
    aggregate,
    average_precision,
    cl_dice,
    evaluate,
    instance_masks_from_labels,
    instance_scores,
    match_instances,
    multiscale_iou,
    pairwise_iou_matrix,
    pixel_scores,
)


def _filament(shape=(128, 128), with_barb=True):
    mask = np.zeros(shape, dtype=bool)
    mask[60:66, 20:100] = True
    if with_barb:
        mask[45:60, 55:58] = True
    return mask


def test_pixel_scores_on_identical_masks():
    mask = _filament()
    scores = pixel_scores(mask, mask)
    assert scores.iou == pytest.approx(1.0)
    assert scores.dice == pytest.approx(1.0)
    assert scores.precision == pytest.approx(1.0)
    assert scores.recall == pytest.approx(1.0)


def test_pixel_scores_on_disjoint_masks():
    a = np.zeros((32, 32), dtype=bool); a[:8, :8] = True
    b = np.zeros((32, 32), dtype=bool); b[20:28, 20:28] = True
    scores = pixel_scores(a, b)
    assert scores.iou == pytest.approx(0.0)
    assert scores.true_positive == 0


def test_pixel_scores_when_both_are_empty():
    empty = np.zeros((16, 16), dtype=bool)
    assert pixel_scores(empty, empty).iou == pytest.approx(1.0)


def test_valid_mask_excludes_off_disk_pixels():
    truth = np.zeros((32, 32), dtype=bool); truth[:4, :4] = True
    prediction = truth.copy(); prediction[28:, 28:] = True  # a false positive off-disk
    valid = np.zeros((32, 32), dtype=bool); valid[:16, :16] = True
    assert pixel_scores(prediction, truth, valid).iou == pytest.approx(1.0)
    assert pixel_scores(prediction, truth).iou < 1.0


def test_cl_dice_punishes_a_deleted_barb_more_than_iou():
    truth = _filament(with_barb=True)
    without = _filament(with_barb=False)
    iou_drop = 1.0 - pixel_scores(without, truth).iou
    cl_drop = 1.0 - cl_dice(without, truth)
    # Losing a barb costs proportionally more topology than area.
    assert cl_drop > iou_drop * 0.8
    assert cl_dice(truth, truth) == pytest.approx(1.0)


def test_cl_dice_edge_cases():
    empty = np.zeros((16, 16), dtype=bool)
    solid = np.ones((16, 16), dtype=bool)
    assert cl_dice(empty, empty) == pytest.approx(1.0)
    assert cl_dice(empty, solid) == pytest.approx(0.0)


def test_multiscale_iou_is_more_forgiving_of_a_small_offset():
    truth = _filament()
    shifted = np.roll(truth, 1, axis=0)
    assert multiscale_iou(truth, truth) == pytest.approx(1.0)
    assert multiscale_iou(shifted, truth) > pixel_scores(shifted, truth).iou


def test_multiscale_iou_still_punishes_a_missing_structure():
    truth = _filament()
    empty = np.zeros_like(truth)
    assert multiscale_iou(empty, truth) == pytest.approx(0.0)


def test_multiscale_iou_curve_rises_with_scale():
    truth = _filament()
    shifted = np.roll(truth, 2, axis=0)
    _, curve = multiscale_iou(shifted, truth, return_curve=True)
    assert curve[64] >= curve[1]


def test_pairwise_iou_matrix_shape_and_values():
    predictions = [np.zeros((32, 32), dtype=bool) for _ in range(2)]
    predictions[0][:8, :8] = True
    predictions[1][20:28, 20:28] = True
    truths = [np.zeros((32, 32), dtype=bool)]
    truths[0][:8, :8] = True

    iou = pairwise_iou_matrix(predictions, truths)
    assert iou.shape == (2, 1)
    assert iou[0, 0] == pytest.approx(1.0)
    assert iou[1, 0] == pytest.approx(0.0)


def test_pairwise_iou_handles_empty_inputs():
    assert pairwise_iou_matrix([], []).shape == (0, 0)


def test_match_instances_is_one_to_one():
    iou = np.array([[0.9, 0.6], [0.7, 0.8]])
    matches, unmatched_pred, unmatched_true = match_instances(iou, 0.5)
    assert len(matches) == 2
    assert not unmatched_pred and not unmatched_true
    assert len({p for p, _ in matches}) == 2
    assert len({t for _, t in matches}) == 2


def test_match_instances_respects_the_threshold():
    iou = np.array([[0.4]])
    matches, unmatched_pred, unmatched_true = match_instances(iou, 0.5)
    assert not matches and unmatched_pred == [0] and unmatched_true == [0]


def test_average_precision_is_one_for_a_perfect_prediction():
    iou = np.eye(3)
    assert average_precision(iou, 0.5) == pytest.approx(1.0)


def test_average_precision_edge_cases():
    assert average_precision(np.zeros((0, 0)), 0.5) == pytest.approx(1.0)
    assert average_precision(np.zeros((2, 0)), 0.5) == pytest.approx(0.0)
    assert average_precision(np.zeros((0, 2)), 0.5) == pytest.approx(0.0)


def test_instance_scores_report_hits_and_misses():
    truths = []
    for offset in (0, 40, 80):
        mask = np.zeros((128, 128), dtype=bool)
        mask[offset : offset + 20, 10:60] = True
        truths.append(mask)
    predictions = truths[:2]  # the third filament is missed

    scores = instance_scores(predictions, truths, match_threshold=0.5)
    assert scores.n_matched == 2
    assert scores.hit_rate == pytest.approx(2 / 3)
    assert scores.miss_rate == pytest.approx(1 / 3)
    assert scores.false_discovery_rate == pytest.approx(0.0)
    assert scores.mean_pairwise_iou == pytest.approx(2 / 3, abs=0.02)


def test_instance_masks_from_labels_skips_background():
    labels = np.array([[0, 1], [2, 2]])
    masks = instance_masks_from_labels(labels)
    assert len(masks) == 2


def test_evaluate_returns_every_headline_metric():
    truth = np.zeros((64, 64), dtype=np.int32)
    truth[10:20, 10:40] = 1
    truth[30:40, 10:40] = 2
    result = evaluate(truth.copy(), truth)
    for key in ("iou", "dice", "cl_dice", "msiou", "hit_rate", "miss_rate",
                "mean_pairwise_iou", "AP@0.50", "mAP"):
        assert key in result
    assert result["iou"] == pytest.approx(1.0)
    assert result["hit_rate"] == pytest.approx(1.0)


def test_aggregate_sums_counts_and_averages_rates():
    records = [
        {"iou": 0.4, "true_positive": 10, "n_truth": 2},
        {"iou": 0.8, "true_positive": 30, "n_truth": 4},
    ]
    merged = aggregate(records)
    assert merged["iou"] == pytest.approx(0.6)
    assert merged["true_positive"] == pytest.approx(40)
    assert merged["n_truth"] == pytest.approx(6)
    assert merged["n_images"] == pytest.approx(2)


def test_aggregate_of_nothing_is_empty():
    assert aggregate([]) == {}
