"""Evaluation metrics for filament segmentation.

The challenge scores submissions with pixel IoU, precision, recall,
``AP@IoU`` at a range of thresholds, hit rate, miss rate and a Multi-scale
Intersection over Union (MSIoU).  All of those are implemented here, plus
clDice as a direct read-out of whether fine structure survived.

A note on MSIoU.  It exists because plain IoU is a poor judge of thin
structures: a filament three pixels wide that is predicted one pixel to the
left scores near zero, even though it is, for any scientific purpose, correct.
MSIoU compares the *edges* of the two masks over a ladder of grid resolutions.
At fine grids it behaves like ordinary IoU; at coarse grids a small offset stops
mattering, while a structure that is missing altogether still scores nothing.
The implementation follows the published description: Sobel edge maps, grid
occupancy at several cell sizes, and aggregation across scales.  Should the
organisers publish reference code with different conventions, replace
:func:`multiscale_iou` -- everything else here is independent of it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage as ndi


# --------------------------------------------------------------------------
# Pixel-level metrics
# --------------------------------------------------------------------------


@dataclass
class PixelScores:
    """Pixel-level agreement between a prediction and the ground truth."""

    iou: float
    dice: float
    precision: float
    recall: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int

    def as_dict(self) -> dict[str, float]:
        return asdict(self)


def pixel_scores(
    prediction: np.ndarray, truth: np.ndarray, valid: np.ndarray | None = None
) -> PixelScores:
    """Confusion-matrix metrics over all pixels.

    Args:
        prediction: Boolean predicted mask.
        truth: Boolean ground-truth mask.
        valid: Optional mask of pixels to score, normally the solar disk.  Any
            prediction outside the disk is meaningless, so restricting to it
            avoids flattering or punishing a model for off-disk noise.
    """
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool)
        prediction = prediction & valid
        truth = truth & valid

    true_positive = int(np.count_nonzero(prediction & truth))
    false_positive = int(np.count_nonzero(prediction & ~truth))
    false_negative = int(np.count_nonzero(~prediction & truth))

    union = true_positive + false_positive + false_negative
    iou = true_positive / union if union else 1.0
    denominator = 2 * true_positive + false_positive + false_negative
    dice = 2 * true_positive / denominator if denominator else 1.0
    precision = (
        true_positive / (true_positive + false_positive)
        if (true_positive + false_positive)
        else 1.0
    )
    recall = (
        true_positive / (true_positive + false_negative)
        if (true_positive + false_negative)
        else 1.0
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    return PixelScores(
        iou, dice, precision, recall, f1, true_positive, false_positive, false_negative
    )


def cl_dice(prediction: np.ndarray, truth: np.ndarray) -> float:
    """Centreline Dice: how much of each mask's skeleton the other mask contains.

    This is the topology counterpart to Dice.  A prediction that keeps every
    filament body but shaves off the barbs loses little Dice and a great deal of
    clDice, which is exactly the failure this challenge penalises.
    """
    from skimage.morphology import skeletonize

    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)
    if not prediction.any() and not truth.any():
        return 1.0
    if not prediction.any() or not truth.any():
        return 0.0

    skeleton_pred = skeletonize(prediction)
    skeleton_true = skeletonize(truth)
    precision = (
        np.count_nonzero(skeleton_pred & truth) / np.count_nonzero(skeleton_pred)
        if skeleton_pred.any()
        else 0.0
    )
    sensitivity = (
        np.count_nonzero(skeleton_true & prediction) / np.count_nonzero(skeleton_true)
        if skeleton_true.any()
        else 0.0
    )
    if precision + sensitivity == 0:
        return 0.0
    return float(2 * precision * sensitivity / (precision + sensitivity))


# --------------------------------------------------------------------------
# Multi-scale IoU
# --------------------------------------------------------------------------


def sobel_edges(mask: np.ndarray) -> np.ndarray:
    """Boolean edge map of a binary mask, via a Sobel gradient."""
    mask = np.asarray(mask, dtype=np.float32)
    if not mask.any():
        return np.zeros(mask.shape, dtype=bool)
    gradient = np.hypot(ndi.sobel(mask, axis=0), ndi.sobel(mask, axis=1))
    return gradient > 0


def _grid_occupancy(binary: np.ndarray, cell: int) -> np.ndarray:
    """Reduce a mask to a grid, marking each cell that contains any True pixel."""
    if cell <= 1:
        return binary
    height, width = binary.shape
    pad_y = (-height) % cell
    pad_x = (-width) % cell
    if pad_y or pad_x:
        binary = np.pad(binary, ((0, pad_y), (0, pad_x)), constant_values=False)
    rows = binary.shape[0] // cell
    cols = binary.shape[1] // cell
    return binary.reshape(rows, cell, cols, cell).any(axis=(1, 3))


def multiscale_iou(
    prediction: np.ndarray,
    truth: np.ndarray,
    scales: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    return_curve: bool = False,
) -> float | tuple[float, dict[int, float]]:
    """Multi-scale IoU over Sobel edge maps.

    At each grid cell size we mark which cells contain an edge of the prediction
    and which contain an edge of the truth, then take the IoU of those two sets
    of cells.  Averaging over the scale ladder rewards a prediction whose
    structure is right even when its exact pixel placement is a little off,
    while still giving nothing for structures that are simply absent.

    Args:
        prediction: Boolean predicted mask.
        truth: Boolean ground-truth mask.
        scales: Grid cell sizes in pixels, from fine to coarse.
        return_curve: Also return the per-scale IoU values.

    Returns:
        The mean IoU across scales, optionally with the per-scale breakdown.
    """
    prediction = np.asarray(prediction, dtype=bool)
    truth = np.asarray(truth, dtype=bool)

    edges_pred = sobel_edges(prediction)
    edges_true = sobel_edges(truth)

    curve: dict[int, float] = {}
    for cell in scales:
        grid_pred = _grid_occupancy(edges_pred, cell)
        grid_true = _grid_occupancy(edges_true, cell)
        intersection = int(np.count_nonzero(grid_pred & grid_true))
        union = int(np.count_nonzero(grid_pred | grid_true))
        curve[int(cell)] = intersection / union if union else 1.0

    score = float(np.mean(list(curve.values()))) if curve else 1.0
    if return_curve:
        return score, curve
    return score


# --------------------------------------------------------------------------
# Instance-level metrics
# --------------------------------------------------------------------------


def instance_masks_from_labels(labels: np.ndarray) -> list[np.ndarray]:
    """Split an integer label map into one boolean mask per label."""
    labels = np.asarray(labels)
    return [labels == value for value in np.unique(labels) if value > 0]


def pairwise_iou_matrix(
    predictions: list[np.ndarray], truths: list[np.ndarray]
) -> np.ndarray:
    """IoU of every predicted instance against every ground-truth instance.

    Computed from flat bincounts over the two label maps rather than by
    comparing every pair directly, so the cost does not blow up on frames with
    many filaments.

    Returns:
        An array of shape ``(len(predictions), len(truths))``.
    """
    if not predictions or not truths:
        return np.zeros((len(predictions), len(truths)), dtype=np.float64)

    shape = truths[0].shape
    pred_labels = np.zeros(shape, dtype=np.int32)
    for index, mask in enumerate(predictions, start=1):
        pred_labels[mask] = index
    true_labels = np.zeros(shape, dtype=np.int32)
    for index, mask in enumerate(truths, start=1):
        true_labels[mask] = index

    n_pred, n_true = len(predictions), len(truths)
    combined = pred_labels.astype(np.int64) * (n_true + 1) + true_labels.astype(np.int64)
    counts = np.bincount(combined.ravel(), minlength=(n_pred + 1) * (n_true + 1))
    table = counts.reshape(n_pred + 1, n_true + 1).astype(np.float64)

    intersection = table[1:, 1:]
    pred_area = table[1:, :].sum(axis=1, keepdims=True)
    true_area = table[:, 1:].sum(axis=0, keepdims=True)
    union = pred_area + true_area - intersection
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(union > 0, intersection / union, 0.0)
    return iou


def match_instances(
    iou: np.ndarray, threshold: float = 0.5
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedily pair predictions with ground truth by descending IoU.

    Greedy matching is used rather than the Hungarian algorithm because the
    challenge metrics are defined per pair and greedy matching is what the usual
    detection evaluators do; the two agree in all but pathological cases.

    Args:
        iou: Matrix from :func:`pairwise_iou_matrix`.
        threshold: Minimum IoU for a pair to count as a match.

    Returns:
        ``(matches, unmatched_predictions, unmatched_truths)`` where ``matches``
        holds ``(prediction_index, truth_index)`` pairs.
    """
    matches: list[tuple[int, int]] = []
    if iou.size:
        candidates = np.argwhere(iou >= threshold)
        order = np.argsort(-iou[candidates[:, 0], candidates[:, 1]]) if len(candidates) else []
        used_pred: set[int] = set()
        used_true: set[int] = set()
        for index in order:
            pred_index, true_index = (int(v) for v in candidates[index])
            if pred_index in used_pred or true_index in used_true:
                continue
            used_pred.add(pred_index)
            used_true.add(true_index)
            matches.append((pred_index, true_index))
    else:
        used_pred, used_true = set(), set()

    unmatched_pred = [i for i in range(iou.shape[0]) if i not in used_pred]
    unmatched_true = [j for j in range(iou.shape[1]) if j not in used_true]
    return matches, unmatched_pred, unmatched_true


def average_precision(
    iou: np.ndarray, threshold: float, scores: np.ndarray | None = None
) -> float:
    """Detection average precision at one IoU threshold.

    Predictions are ranked by ``scores`` (or by area, if no scores are given),
    matched greedily to unclaimed ground truth, and the resulting
    precision-recall curve is integrated with the all-point interpolation used
    by COCO.
    """
    n_pred, n_true = iou.shape
    if n_true == 0:
        return 1.0 if n_pred == 0 else 0.0
    if n_pred == 0:
        return 0.0

    order = np.argsort(-scores) if scores is not None else np.arange(n_pred)
    claimed = np.zeros(n_true, dtype=bool)
    true_positives = np.zeros(n_pred, dtype=np.float64)
    false_positives = np.zeros(n_pred, dtype=np.float64)

    for rank, pred_index in enumerate(order):
        row = iou[pred_index].copy()
        row[claimed] = -1.0
        best = int(np.argmax(row))
        if row[best] >= threshold:
            claimed[best] = True
            true_positives[rank] = 1.0
        else:
            false_positives[rank] = 1.0

    cumulative_tp = np.cumsum(true_positives)
    cumulative_fp = np.cumsum(false_positives)
    recall = cumulative_tp / n_true
    precision = cumulative_tp / np.maximum(cumulative_tp + cumulative_fp, 1e-9)

    # All-point interpolation: make precision monotonically decreasing, then
    # integrate over recall.
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([precision[0] if precision.size else 0.0], precision))
    return float(np.sum(np.diff(recall) * precision[1:]))


@dataclass
class InstanceScores:
    """Instance-level agreement for one observation."""

    n_predicted: int
    n_truth: int
    n_matched: int
    hit_rate: float
    miss_rate: float
    false_discovery_rate: float
    mean_pairwise_iou: float
    average_precision: dict[float, float]

    def as_dict(self) -> dict[str, float]:
        flat: dict[str, float] = {
            "n_predicted": self.n_predicted,
            "n_truth": self.n_truth,
            "n_matched": self.n_matched,
            "hit_rate": self.hit_rate,
            "miss_rate": self.miss_rate,
            "false_discovery_rate": self.false_discovery_rate,
            "mean_pairwise_iou": self.mean_pairwise_iou,
        }
        for threshold, value in self.average_precision.items():
            flat[f"AP@{threshold:.2f}"] = value
        if self.average_precision:
            flat["mAP"] = float(np.mean(list(self.average_precision.values())))
        return flat


def instance_scores(
    predictions: list[np.ndarray],
    truths: list[np.ndarray],
    match_threshold: float = 0.5,
    ap_thresholds: tuple[float, ...] = (0.25, 0.50, 0.75),
    scores: np.ndarray | None = None,
) -> InstanceScores:
    """Instance-level metrics: hit rate, miss rate and AP at several thresholds.

    ``mean_pairwise_iou`` is the challenge's pairwise IoU: for each ground-truth
    filament, the IoU with its best-overlapping prediction, counting an
    unmatched filament as zero.  Averaging over ground truth rather than over
    matches is deliberate -- otherwise a model could score well by predicting
    one tiny, perfect filament and ignoring the rest.
    """
    iou = pairwise_iou_matrix(predictions, truths)
    matches, unmatched_pred, _ = match_instances(iou, match_threshold)

    n_truth = len(truths)
    n_pred = len(predictions)
    hit_rate = len(matches) / n_truth if n_truth else 1.0
    fdr = len(unmatched_pred) / n_pred if n_pred else 0.0

    if n_truth == 0:
        mean_pairwise = 1.0 if n_pred == 0 else 0.0
    elif n_pred == 0:
        mean_pairwise = 0.0
    else:
        mean_pairwise = float(np.mean(iou.max(axis=0)))

    return InstanceScores(
        n_predicted=n_pred,
        n_truth=n_truth,
        n_matched=len(matches),
        hit_rate=hit_rate,
        miss_rate=1.0 - hit_rate,
        false_discovery_rate=fdr,
        mean_pairwise_iou=mean_pairwise,
        average_precision={
            float(t): average_precision(iou, float(t), scores) for t in ap_thresholds
        },
    )


def evaluate(
    predicted_labels: np.ndarray,
    truth_labels: np.ndarray,
    valid: np.ndarray | None = None,
    match_threshold: float = 0.5,
    ap_thresholds: tuple[float, ...] = (0.25, 0.50, 0.75),
) -> dict[str, float]:
    """Score one observation on every metric the challenge uses.

    Args:
        predicted_labels: Integer instance label map of the prediction.
        truth_labels: Integer instance label map of the ground truth.
        valid: Optional on-disk mask.
        match_threshold: IoU above which an instance counts as detected.
        ap_thresholds: IoU thresholds at which to report average precision.

    Returns:
        A flat dictionary of metric name to value.
    """
    predicted_labels = np.asarray(predicted_labels)
    truth_labels = np.asarray(truth_labels)

    pred_mask = predicted_labels > 0
    true_mask = truth_labels > 0
    if valid is not None:
        pred_mask = pred_mask & valid
        true_mask = true_mask & valid

    results = pixel_scores(pred_mask, true_mask).as_dict()
    results["cl_dice"] = cl_dice(pred_mask, true_mask)
    results["msiou"] = float(multiscale_iou(pred_mask, true_mask))

    predictions = instance_masks_from_labels(predicted_labels)
    truths = instance_masks_from_labels(truth_labels)
    areas = np.array([float(m.sum()) for m in predictions]) if predictions else None
    results.update(
        instance_scores(
            predictions, truths, match_threshold, ap_thresholds, areas
        ).as_dict()
    )
    # The challenge ranks on Panoptic Quality and reports the fragmentation and
    # over-merging behind it, so both are always computed.
    results.update(panoptic_quality(predictions, truths, match_threshold).as_dict())
    results.update(fragmentation(predictions, truths).as_dict())
    return results


def aggregate(per_image: list[dict[str, float]]) -> dict[str, float]:
    """Average a list of per-observation metric dictionaries.

    Counts (``true_positive``, ``n_truth`` and so on) are summed; everything
    else is averaged, so the headline numbers are per-observation means.
    """
    if not per_image:
        return {}
    summed = {"true_positive", "false_positive", "false_negative",
              "n_predicted", "n_truth", "n_matched",
              "pq_tp", "pq_fp", "pq_fn",
              "one_to_one", "one_to_many", "many_to_one", "missed", "spurious"}
    keys = sorted({key for record in per_image for key in record})
    out: dict[str, float] = {}
    for key in keys:
        values = [record[key] for record in per_image if key in record]
        if not values:
            continue
        out[key] = float(np.sum(values)) if key in summed else float(np.mean(values))
    out["n_images"] = float(len(per_image))
    return out


# --------------------------------------------------------------------------
# Panoptic Quality, and the fragmentation it is meant to expose
# --------------------------------------------------------------------------


@dataclass
class PanopticScores:
    """Panoptic Quality and its two factors, for one observation."""

    pq: float
    """Panoptic Quality: segmentation quality times recognition quality."""
    sq: float
    """Segmentation Quality: mean IoU over matched pairs."""
    rq: float
    """Recognition Quality: the F1 of the matching itself."""
    true_positive: int
    false_positive: int
    false_negative: int

    def as_dict(self) -> dict[str, float]:
        return {
            "pq": self.pq,
            "sq": self.sq,
            "rq": self.rq,
            "pq_tp": self.true_positive,
            "pq_fp": self.false_positive,
            "pq_fn": self.false_negative,
        }


def panoptic_quality(
    predictions: list[np.ndarray],
    truths: list[np.ndarray],
    threshold: float = 0.5,
) -> PanopticScores:
    """Panoptic Quality, as defined by Kirillov et al. (CVPR 2019).

    .. math::

        PQ = \\frac{\\sum_{(y, \\hat{y}) \\in TP} IoU(y, \\hat{y})}
                  {|TP| + \\tfrac{1}{2}|FP| + \\tfrac{1}{2}|FN|}

    A predicted segment and a ground-truth segment match when their IoU exceeds
    ``threshold``. At the standard 0.5 that matching is provably unique -- no
    segment can overlap two others by more than half -- so no tie-breaking is
    needed and the result does not depend on the order segments are considered
    in.

    PQ is unforgiving in exactly the way this task needs. Splitting one filament
    into two predictions yields one match and one false positive; merging two
    filaments into one yields one match and one false negative. Both cost the
    same as missing a filament outright, which is why a segmentation can score
    well on pixel IoU and badly here.

    Args:
        predictions: One boolean mask per predicted filament.
        truths: One boolean mask per annotated filament.
        threshold: IoU above which a pair counts as matched.

    Returns:
        The :class:`PanopticScores` for this observation.
    """
    if not predictions and not truths:
        return PanopticScores(1.0, 1.0, 1.0, 0, 0, 0)
    if not predictions or not truths:
        return PanopticScores(
            0.0, 0.0, 0.0, 0, len(predictions), len(truths)
        )

    iou = pairwise_iou_matrix(predictions, truths)
    matched_pred, matched_true = np.nonzero(iou > threshold)
    matched_iou = iou[matched_pred, matched_true]

    true_positive = int(matched_iou.size)
    false_positive = len(predictions) - true_positive
    false_negative = len(truths) - true_positive

    denominator = true_positive + 0.5 * false_positive + 0.5 * false_negative
    pq = float(matched_iou.sum() / denominator) if denominator > 0 else 0.0
    sq = float(matched_iou.mean()) if true_positive else 0.0
    rq = float(true_positive / denominator) if denominator > 0 else 0.0
    return PanopticScores(pq, sq, rq, true_positive, false_positive, false_negative)


@dataclass
class FragmentationScores:
    """How prediction and truth segments correspond, beyond one-to-one."""

    one_to_one: int
    one_to_many: int
    """Ground-truth filaments broken across several predictions (fragmentation)."""
    many_to_one: int
    """Predictions covering several ground-truth filaments (over-merging)."""
    missed: int
    spurious: int
    fragments_per_split: float
    """Mean number of predictions covering a fragmented filament."""

    def as_dict(self) -> dict[str, float]:
        return {
            "one_to_one": self.one_to_one,
            "one_to_many": self.one_to_many,
            "many_to_one": self.many_to_one,
            "missed": self.missed,
            "spurious": self.spurious,
            "fragments_per_split": self.fragments_per_split,
        }


def fragmentation(
    predictions: list[np.ndarray],
    truths: list[np.ndarray],
    overlap: float = 0.1,
) -> FragmentationScores:
    """Count how often filaments are split apart or welded together.

    Panoptic Quality punishes both faults but does not say which occurred. This
    separates them, which is what tells you whether to merge more aggressively
    or less.

    A prediction and a truth are taken to correspond when they overlap by more
    than ``overlap`` of the smaller of the two. That is deliberately looser than
    the matching threshold: a filament split into three pieces has no piece
    reaching IoU 0.5, and the point here is to notice precisely that.

    Args:
        predictions: One boolean mask per predicted filament.
        truths: One boolean mask per annotated filament.
        overlap: Fraction of the smaller segment that must overlap.

    Returns:
        The :class:`FragmentationScores` for this observation.
    """
    if not truths:
        return FragmentationScores(0, 0, 0, 0, len(predictions), 0.0)
    if not predictions:
        return FragmentationScores(0, 0, 0, len(truths), 0, 0.0)

    shape = truths[0].shape
    pred_labels = np.zeros(shape, dtype=np.int32)
    for index, mask in enumerate(predictions, start=1):
        pred_labels[mask] = index
    true_labels = np.zeros(shape, dtype=np.int32)
    for index, mask in enumerate(truths, start=1):
        true_labels[mask] = index

    n_pred, n_true = len(predictions), len(truths)
    combined = pred_labels.astype(np.int64) * (n_true + 1) + true_labels.astype(np.int64)
    table = np.bincount(
        combined.ravel(), minlength=(n_pred + 1) * (n_true + 1)
    ).reshape(n_pred + 1, n_true + 1).astype(np.float64)

    intersection = table[1:, 1:]
    pred_area = table[1:, :].sum(axis=1, keepdims=True)
    true_area = table[:, 1:].sum(axis=0, keepdims=True)
    smaller = np.minimum(pred_area, true_area)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(smaller > 0, intersection / smaller, 0.0)
    linked = share > overlap

    per_truth = linked.sum(axis=0)      # predictions touching each true filament
    per_pred = linked.sum(axis=1)       # true filaments touched by each prediction

    missed = int((per_truth == 0).sum())
    spurious = int((per_pred == 0).sum())
    one_to_many = int((per_truth > 1).sum())
    many_to_one = int((per_pred > 1).sum())
    # A clean one-to-one correspondence: the truth is touched by exactly one
    # prediction, and that prediction touches exactly this truth and no other.
    single = per_truth == 1
    if single.any():
        partner = np.asarray(linked[:, single]).argmax(axis=0)
        one_to_one = int((per_pred[partner] == 1).sum())
    else:
        one_to_one = 0
    split_counts = per_truth[per_truth > 1]
    return FragmentationScores(
        one_to_one=one_to_one,
        one_to_many=one_to_many,
        many_to_one=many_to_one,
        missed=missed,
        spurious=spurious,
        fragments_per_split=float(split_counts.mean()) if split_counts.size else 0.0,
    )
