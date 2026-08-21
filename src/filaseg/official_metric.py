"""Organizer-style instance evaluation for the filament challenge.

This module follows the public Self_Evaluation_Notebook semantics used by the
Solar Filament Segmentation Challenge 2026:

* score each annotator-image record against all predictions for that physical
  image;
* preserve ground-truth instances as independent masks (do not collapse them
  into an integer label map, which can erase overlapping pixels);
* a hit is any GT/prediction pair with IoU strictly greater than 0.5;
* TP is the number of hit pairs; predictions with no hit are FP; truths with no
  hit are FN;
* aggregate TP IoU, FP and FN across the complete evaluation set before forming
  PQ, SQ and RQ.

The evaluator is independent of model and post-processing code so it can be
regression-tested and reused by training, tuning and submission analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .metrics import pairwise_iou_matrix

IOU_THRESHOLD = 0.5


@dataclass
class OfficialPQResult:
    """Globally accumulated organizer-style PQ statistics."""

    pq: float
    sq: float
    rq: float
    tp: int
    fp: int
    fn: int
    n_records: int
    tp_ious: list[float] = field(default_factory=list)
    pair_ious: list[float] = field(default_factory=list)
    pair_dices: list[float] = field(default_factory=list)

    def as_dict(self) -> dict[str, float]:
        return {
            "official_pq": self.pq,
            "official_sq": self.sq,
            "official_rq": self.rq,
            "official_tp": float(self.tp),
            "official_fp": float(self.fp),
            "official_fn": float(self.fn),
            "official_n_records": float(self.n_records),
            "official_mean_tp_iou": self.sq,
            "official_mean_pair_iou": (
                float(np.mean(self.pair_ious)) if self.pair_ious else 0.0
            ),
            "official_mean_pair_dice": (
                float(np.mean(self.pair_dices)) if self.pair_dices else 0.0
            ),
        }


@dataclass
class OfficialPQAccumulator:
    """Streaming evaluator with global PQ aggregation."""

    threshold: float = IOU_THRESHOLD
    tp_ious: list[float] = field(default_factory=list)
    pair_ious: list[float] = field(default_factory=list)
    pair_dices: list[float] = field(default_factory=list)
    fp: int = 0
    fn: int = 0
    n_records: int = 0

    def add(
        self,
        truths: list[np.ndarray],
        predictions: list[np.ndarray],
    ) -> None:
        """Add one annotator-image record using independent instance masks."""
        self.n_records += 1
        n_truth = len(truths)
        n_pred = len(predictions)

        if n_truth == 0:
            self.fp += n_pred
            return
        if n_pred == 0:
            self.fn += n_truth
            return

        # pairwise_iou_matrix returns (n_pred, n_truth).
        iou = pairwise_iou_matrix(predictions, truths)
        hit = iou > float(self.threshold)

        self.tp_ious.extend(iou[hit].astype(float).tolist())
        pred_hits = hit.sum(axis=1)
        truth_hits = hit.sum(axis=0)
        self.fp += int(np.count_nonzero(pred_hits == 0))
        self.fn += int(np.count_nonzero(truth_hits == 0))

        overlapping = iou > 0
        overlaps = iou[overlapping].astype(np.float64)
        if overlaps.size:
            self.pair_ious.extend(overlaps.tolist())
            self.pair_dices.extend((2.0 * overlaps / (1.0 + overlaps)).tolist())

    def result(self) -> OfficialPQResult:
        tp = len(self.tp_ious)
        denominator = tp + 0.5 * self.fp + 0.5 * self.fn
        pq = float(np.sum(self.tp_ious) / denominator) if denominator > 0 else 0.0
        sq = float(np.mean(self.tp_ious)) if tp else 0.0
        rq = float(tp / denominator) if denominator > 0 else 0.0
        return OfficialPQResult(
            pq=pq,
            sq=sq,
            rq=rq,
            tp=tp,
            fp=self.fp,
            fn=self.fn,
            n_records=self.n_records,
            tp_ious=list(self.tp_ious),
            pair_ious=list(self.pair_ious),
            pair_dices=list(self.pair_dices),
        )


def evaluate_official_pq(
    rows: list[tuple[list[np.ndarray], list[np.ndarray]]],
    threshold: float = IOU_THRESHOLD,
) -> OfficialPQResult:
    """Batch convenience wrapper around :class:`OfficialPQAccumulator`."""
    accumulator = OfficialPQAccumulator(threshold=threshold)
    for truths, predictions in rows:
        accumulator.add(truths, predictions)
    return accumulator.result()
