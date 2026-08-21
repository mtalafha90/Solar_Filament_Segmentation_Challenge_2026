#!/usr/bin/env python3
"""Tune post-processing with organizer-style globally accumulated PQ.

This is intentionally separate from ``tune_postprocess.py`` so historical
matched-Dice experiments remain reproducible. It reuses that script's grouped
validation split and probability-cache implementation, but scores each
annotator-image record against the original independent GT instance masks and
forms PQ only after globally accumulating TP IoU, FP and FN.

Example comparing the original E20 post-processing with Submission 2::

    python scripts/tune_official_pq.py --data-dir data \
        --checkpoint runs/cpu_filanet_20epoch/best.pt --limit 0 \
        --thresholds 0.93 --merge-gap 18 40 \
        --min-area-fraction 0.00012 0.00023
"""

from __future__ import annotations

import argparse
import itertools
import json
import warnings
from pathlib import Path

import _bootstrap  # noqa: F401
import numpy as np

import tune_postprocess as legacy
from filaseg.data.layout import discover, resolve_annotations
from filaseg.metrics import fragmentation, instance_dice, instance_masks_from_labels, pixel_scores
from filaseg.official_metric import OfficialPQAccumulator
from filaseg.postprocess.instances import InstanceConfig, extract_instances


def _grid(args: argparse.Namespace) -> list[tuple[float, float, float, float]]:
    out: list[tuple[float, float, float, float]] = []
    for threshold, confidence, gap, area in itertools.product(
        args.thresholds, args.min_confidence, args.merge_gap, args.min_area_fraction
    ):
        if confidence > 0 and confidence <= threshold:
            continue
        out.append((float(threshold), float(confidence), float(gap), float(area)))
    return out


def _truth_masks(dataset, index: int, prepared) -> list[np.ndarray]:
    """Return original independent annotation masks, preserving overlaps."""
    record = dataset.records[index]
    masks = [np.asarray(mask, dtype=bool) for mask in record.instance_masks()]
    target_shape = prepared.mask.shape
    if any(mask.shape != target_shape for mask in masks):
        raise ValueError(
            f"{record.file_name}: GT instance mask shape differs from prepared image "
            f"shape {target_shape}; load/rescale must happen before official scoring"
        )
    return masks


def score_config(dataset, indices, paths, config: InstanceConfig) -> dict[str, float]:
    """Score one configuration with official global PQ plus diagnostics."""
    official = OfficialPQAccumulator()
    matched_dice: list[float] = []
    paired_dice: list[float] = []
    foreground_dice: list[float] = []
    instance_counts: list[float] = []
    one_to_many = many_to_one = missed = spurious = 0

    for index, path in zip(indices, paths):
        prepared = dataset[index]
        probability = np.load(path).astype(np.float32)
        labels = extract_instances(probability, prepared.valid, config)
        predicted = instance_masks_from_labels(labels)
        truths = _truth_masks(dataset, index, prepared)

        official.add(truths, predicted)

        dice = instance_dice(predicted, truths)
        matched_dice.append(float(dice.matched_dice))
        paired_dice.append(float(dice.mean_paired_dice))

        truth_union = np.zeros(prepared.mask.shape, dtype=bool)
        for truth in truths:
            truth_union |= truth
        foreground_dice.append(
            float(pixel_scores(labels > 0, truth_union, prepared.valid).dice)
        )
        instance_counts.append(float(len(predicted)))

        frag = fragmentation(predicted, truths)
        one_to_many += frag.one_to_many
        many_to_one += frag.many_to_one
        missed += frag.missed
        spurious += frag.spurious

    result = official.result().as_dict()
    result.update(
        {
            "matched_dice_mean": float(np.mean(matched_dice)) if matched_dice else 0.0,
            "mean_paired_dice": float(np.mean(paired_dice)) if paired_dice else 0.0,
            "foreground_dice": float(np.mean(foreground_dice)) if foreground_dice else 0.0,
            "n_instances_mean": float(np.mean(instance_counts)) if instance_counts else 0.0,
            "one_to_many": float(one_to_many),
            "many_to_one": float(many_to_one),
            "missed": float(missed),
            "spurious": float(spurious),
        }
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"), dest="data_dir")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--image-dir", type=Path, dest="image_dir")
    parser.add_argument("--cache-dir", type=Path, default=Path("data/cache"), dest="cache_dir")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prob-cache", type=Path, default=Path("runs/prob_cache"), dest="prob_cache")
    parser.add_argument("--limit", type=int, default=0, help="validation records; 0 = full split")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tile-size", type=int, default=None, dest="tile_size")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--thresholds", type=float, nargs="+", default=[0.93])
    parser.add_argument("--min-confidence", type=float, nargs="+", dest="min_confidence", default=[0.0])
    parser.add_argument("--merge-gap", type=float, nargs="+", dest="merge_gap", default=[18.0, 40.0])
    parser.add_argument(
        "--min-area-fraction", type=float, nargs="+", dest="min_area_fraction",
        default=[1.2e-4, 2.3e-4],
    )
    args = parser.parse_args()

    if args.data_dir and not args.image_dir:
        args.image_dir = discover(args.data_dir).train_dir
    annotations = resolve_annotations(args.annotations, args.image_dir, args.data_dir)
    blob = legacy._checkpoint_blob(args.checkpoint)
    checkpoint_config = blob.get("config", {}) or {}
    if args.tile_size is None:
        args.tile_size = int(checkpoint_config.get("val_tile", 512))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset, indices, split_info = legacy.validation_dataset(
            annotations,
            args.image_dir,
            args.cache_dir,
            checkpoint_config,
            args.limit,
        )

    print(
        f"official-PQ validation: {split_info['selected_records']} records / "
        f"{split_info['selected_groups']} physical images "
        f"(full={split_info['full_validation_records']}/{split_info['full_validation_groups']})"
    )
    print(f"inference: tile={args.tile_size} tta={args.tta} device={args.device}")
    paths = legacy.cache_probabilities(
        args.checkpoint,
        dataset,
        indices,
        args.prob_cache,
        args.device,
        args.tile_size,
        args.tta,
    )

    grid = _grid(args)
    print(f"\nsweeping {len(grid)} configurations with organizer-style global PQ")
    print(
        f"{'thr':>7} {'gap':>6} {'minarea':>10} {'PQ':>8} {'SQ':>8} {'RQ':>8} "
        f"{'TP':>6} {'FP':>6} {'FN':>6} {'mDice':>8} {'fgDice':>8} {'inst':>7}"
    )

    rows: list[dict[str, float]] = []
    for threshold, confidence, gap, area in grid:
        config = InstanceConfig(
            threshold=threshold,
            min_confidence=confidence,
            merge_gap=gap,
            min_area_fraction=area,
        )
        metrics = score_config(dataset, indices, paths, config)
        row = {
            "threshold": threshold,
            "min_confidence": confidence,
            "merge_gap": gap,
            "min_area_fraction": area,
            **metrics,
        }
        rows.append(row)
        print(
            f"{threshold:7.3f} {gap:6.1f} {area:10.2e} "
            f"{metrics['official_pq']:8.4f} {metrics['official_sq']:8.4f} "
            f"{metrics['official_rq']:8.4f} {metrics['official_tp']:6.0f} "
            f"{metrics['official_fp']:6.0f} {metrics['official_fn']:6.0f} "
            f"{metrics['matched_dice_mean']:8.4f} {metrics['foreground_dice']:8.4f} "
            f"{metrics['n_instances_mean']:7.2f}"
        )

    best = max(rows, key=lambda row: row["official_pq"])
    print("\n" + "=" * 74)
    print("BEST by organizer-style official_pq")
    print("=" * 74)
    for key in (
        "threshold", "min_confidence", "merge_gap", "min_area_fraction",
        "official_pq", "official_sq", "official_rq", "official_tp",
        "official_fp", "official_fn", "matched_dice_mean", "mean_paired_dice",
        "foreground_dice", "n_instances_mean", "spurious", "missed",
        "one_to_many", "many_to_one",
    ):
        print(f"  {key:24s} {best[key]}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "metric_semantics": {
                "match": "IoU > 0.5 (strict)",
                "aggregation": "global TP-IoU/FP/FN over annotation records",
                "ground_truth": "original independent annotation masks; overlaps preserved",
            },
            "checkpoint": str(args.checkpoint),
            "tile_size": int(args.tile_size),
            "tta": bool(args.tta),
            "split": split_info,
            "best": best,
            "results": rows,
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
