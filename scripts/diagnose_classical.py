#!/usr/bin/env python3
"""Work out why the classical detector is scoring badly on a dataset.

The detector is a score map followed by a threshold, so a poor result has
exactly two possible causes, and they call for opposite fixes:

1. **The score map does not separate filaments from everything else.** No
   threshold can rescue that; the preprocessing or the score itself is wrong for
   this data.
2. **The score map separates fine, but the threshold is in the wrong place.**
   That is a calibration problem, fixed with ``scripts/tune_classical.py``.

This script distinguishes them. It reports how well the score ranks true
filament pixels above everything else (the ROC area, which is threshold-free),
then shows where the highest-scoring pixels actually fall, and how filaments and
predictions are distributed across the disk.

Example::

    python scripts/diagnose_classical.py --data-dir data --limit 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from filaseg.classical import (
    ClassicalConfig,
    choose_thresholds,
    hysteresis,
    intensity_deficit,
    ridge_response,
    score_map,
)
from filaseg.data.dataset import MagfiloDataset
from filaseg.data.layout import discover, resolve_annotations
from filaseg.metrics import evaluate
from filaseg.postprocess.instances import InstanceConfig, extract_instances


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Area under the ROC curve, via rank statistics. 0.5 is chance."""
    positives = int(labels.sum())
    negatives = int(labels.size - positives)
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.float64)
    ranks[order] = np.arange(1, scores.size + 1)
    return float((ranks[labels].sum() - positives * (positives + 1) / 2) /
                 (positives * negatives))


def radial_profile(mask: np.ndarray, radius_map: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Fraction of ``mask`` falling in each radial band."""
    total = max(int(mask.sum()), 1)
    return np.array(
        [
            float(((radius_map >= lo) & (radius_map < hi) & mask).sum()) / total
            for lo, hi in zip(edges[:-1], edges[1:])
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, dest="data_dir", default=Path("data"))
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--image-dir", type=Path, dest="image_dir")
    parser.add_argument("--cache-dir", type=Path, dest="cache_dir")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    if args.data_dir and not args.image_dir:
        args.image_dir = discover(args.data_dir).train_dir
    annotations = resolve_annotations(args.annotations, args.image_dir, args.data_dir)

    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = MagfiloDataset(annotations, args.image_dir, args.cache_dir)

    n = min(len(dataset), args.limit)
    print(f"diagnosing on {n} observation(s)\n")

    config = ClassicalConfig()
    edges = np.array([0.0, 0.3, 0.5, 0.7, 0.85, 0.93, 0.97, 1.0])

    aucs, ridge_aucs, deficit_aucs = [], [], []
    contrasts, precisions_at_truth = [], []
    truth_profiles, prediction_profiles, score_profiles = [], [], []
    report_rows = []

    for index in range(n):
        prepared = dataset[index]
        valid = prepared.valid
        truth = prepared.mask & valid
        if truth.sum() == 0:
            continue

        radius_map = prepared.disk.radial_map(prepared.image.shape)
        score = score_map(prepared.image, valid, config)
        ridge = ridge_response(prepared.image, config.scales)
        deficit = intensity_deficit(prepared.image, valid, config.background_scale)

        inside = valid
        labels = truth[inside]
        aucs.append(roc_auc(score[inside], labels))
        ridge_aucs.append(roc_auc(ridge[inside], labels))
        deficit_aucs.append(roc_auc(deficit[inside], labels))

        # How far above the quiet Sun does a filament sit, in the preprocessed
        # image? Filaments are bright there, because preprocessing inverts.
        quiet = prepared.image[valid & ~truth]
        contrasts.append(
            float((prepared.image[truth].mean() - quiet.mean()) / (quiet.std() + 1e-9))
        )

        # Of the pixels the detector would seed on, how many are real filaments?
        n_seed = max(1, int(config.expected_coverage * valid.sum()))
        flat_scores = score[inside]
        cut = np.partition(flat_scores, -n_seed)[-n_seed]
        top = score >= cut
        precisions_at_truth.append(float((top & truth).sum() / max(top.sum(), 1)))

        low, high = choose_thresholds(score[inside], config)
        predicted = extract_instances(
            hysteresis(score, low, high, valid).astype(np.float32),
            valid,
            InstanceConfig(reject_round=True),
        )

        truth_profiles.append(radial_profile(truth, radius_map, edges))
        prediction_profiles.append(radial_profile(predicted > 0, radius_map, edges))
        score_profiles.append(radial_profile(top, radius_map, edges))

        scores = evaluate(predicted, prepared.instances, valid)
        report_rows.append(
            {
                "image_id": str(prepared.image_id),
                "auc": aucs[-1],
                "contrast_sigma": contrasts[-1],
                "seed_precision": precisions_at_truth[-1],
                "iou": scores["iou"],
                "truth_coverage": float(truth.sum() / valid.sum()),
            }
        )

    if not report_rows:
        raise SystemExit("no observation carried any annotated filament")

    print("=" * 70)
    print("IS THE SCORE MAP DISCRIMINATIVE? (threshold-free)")
    print("=" * 70)
    print("  ROC area, filament pixels against everything else on the disk.")
    print("  1.0 is perfect, 0.5 is chance.\n")
    print(f"    combined score   {np.nanmean(aucs):.4f}")
    print(f"    ridge term       {np.nanmean(ridge_aucs):.4f}")
    print(f"    intensity term   {np.nanmean(deficit_aucs):.4f}")
    print(f"\n  filament contrast: {np.mean(contrasts):+.2f} sigma above quiet Sun")
    print("  (must be positive: preprocessing inverts, so filaments end bright)")

    mean_auc = float(np.nanmean(aucs))
    print()
    if mean_auc < 0.6:
        print("  VERDICT: the score map barely separates filaments at all.")
        print("  No threshold can fix this. The preprocessing or the score itself")
        print("  is wrong for this data -- check the contrast line above first:")
        print("  if it is negative or near zero, the polarity or the flattening")
        print("  is the problem, not the threshold.")
    elif mean_auc < 0.85:
        print("  VERDICT: weak but real separation. Tuning will help; expect")
        print("  modest scores. The neural model is the answer here.")
    else:
        print("  VERDICT: the score map separates filaments well. A poor result")
        print("  is a threshold problem -- run scripts/tune_classical.py.")

    print("\n" + "=" * 70)
    print("WHERE DOES IT FIRE? (fraction of pixels per radial band)")
    print("=" * 70)
    header = "  " + "".join(f"{lo:.2f}-{hi:.2f}".rjust(11)
                            for lo, hi in zip(edges[:-1], edges[1:]))
    print(f"  {'band (r/R)':<14}" + header[2:])
    for name, profiles in (
        ("true filaments", truth_profiles),
        ("top-scoring px", score_profiles),
        ("predictions", prediction_profiles),
    ):
        row = np.mean(profiles, axis=0)
        print(f"  {name:<14}" + "".join(f"{v:>11.3f}" for v in row))

    truth_row = np.mean(truth_profiles, axis=0)
    score_row = np.mean(score_profiles, axis=0)
    limb = slice(-2, None)
    if score_row[limb].sum() > 3 * max(truth_row[limb].sum(), 1e-6):
        print("\n  WARNING: the highest-scoring pixels pile up near the limb where")
        print("  few filaments are annotated. Lower disk_fraction in preprocess()")
        print("  so the outer annulus is excluded.")

    print(f"\n  seed precision: {np.mean(precisions_at_truth):.3f}")
    print("  (of the pixels the detector seeds on, the share that are filament)")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(
                {
                    "mean_auc": mean_auc,
                    "mean_contrast_sigma": float(np.mean(contrasts)),
                    "mean_seed_precision": float(np.mean(precisions_at_truth)),
                    "radial_bands": edges.tolist(),
                    "truth_profile": truth_row.tolist(),
                    "score_profile": score_row.tolist(),
                    "prediction_profile": np.mean(prediction_profiles, axis=0).tolist(),
                    "per_image": report_rows,
                },
                handle,
                indent=2,
            )
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
