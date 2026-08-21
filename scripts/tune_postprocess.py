#!/usr/bin/env python3
"""Tune instance construction on a fixed checkpoint, without retraining.

The leaderboard matches predictions to ground truth one filament at a time, so
a model can cover the right pixels and still score badly by splitting filaments
apart or inventing extra ones. That failure lives entirely in the step between
the probability map and the instance labels, which means it can be fixed
without touching the network.

Neural inference is the expensive part and does not change while
post-processing is tuned, so probability maps are computed once, cached to
disk, and reused for every setting in the sweep. A sweep of fifty
configurations then costs one inference pass rather than fifty.

Example::

    python scripts/tune_postprocess.py --data-dir data \\
        --checkpoint runs/filanet/best.pt --limit 60
"""

from __future__ import annotations

import argparse
import itertools
import json
import time
import warnings
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from filaseg.data.dataset import MagfiloDataset
from filaseg.data.layout import discover, resolve_annotations
from filaseg.metrics import (
    fragmentation,
    instance_dice,
    instance_masks_from_labels,
    panoptic_quality,
    pixel_scores,
)
from filaseg.postprocess.instances import InstanceConfig, extract_instances


def cache_probabilities(
    checkpoint: Path,
    dataset: MagfiloDataset,
    indices: list[int],
    cache_dir: Path,
    device: str,
    tile_size: int,
    tta: bool,
) -> list[Path]:
    """Run the network once over the held-out set and cache the probability maps.

    Stored as float16, which is finer than the 8-bit images they came from and
    halves the disk cost. Existing files are reused, so the sweep can be rerun
    or extended without paying for inference again.
    """
    from filaseg.inference import InferenceConfig, predict_probability
    from filaseg.train import load_model

    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = [cache_dir / f"{index:05d}.npy" for index in indices]
    missing = [(i, p) for i, p in zip(indices, paths) if not p.exists()]
    if not missing:
        print(f"reusing {len(paths)} cached probability maps from {cache_dir}")
        return paths

    model, _ = load_model(checkpoint, device)
    config = InferenceConfig(tile_size=tile_size, tta=tta, device=device)
    started = time.time()
    for position, (index, path) in enumerate(missing, start=1):
        prepared = dataset[index]
        probability = predict_probability(model, prepared.input_stack(), config)
        np.save(path, (probability * prepared.valid).astype(np.float16))
        if position % 10 == 0 or position == len(missing):
            print(f"  inference {position}/{len(missing)}  "
                  f"({time.time() - started:.0f}s)", flush=True)
    return paths


def score(
    dataset: MagfiloDataset,
    indices: list[int],
    paths: list[Path],
    config: InstanceConfig,
) -> dict[str, float]:
    """Score one post-processing configuration over the cached maps."""
    rows: list[dict[str, float]] = []
    for index, path in zip(indices, paths):
        prepared = dataset[index]
        probability = np.load(path).astype(np.float32)
        labels = extract_instances(probability, prepared.valid, config)

        predicted = instance_masks_from_labels(labels)
        truths = instance_masks_from_labels(prepared.instances)
        row = instance_dice(predicted, truths).as_dict()
        row.update(panoptic_quality(predicted, truths).as_dict())
        row.update(fragmentation(predicted, truths).as_dict())
        row["foreground_dice"] = pixel_scores(
            labels > 0, prepared.mask, prepared.valid
        ).dice
        row["n_instances"] = float(labels.max())
        rows.append(row)

    summed = {"one_to_many", "many_to_one", "missed", "spurious",
              "pq_tp", "pq_fp", "pq_fn"}
    out: dict[str, float] = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        out[key] = float(np.sum(values)) if key in summed else float(np.mean(values))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, dest="data_dir", default=Path("data"))
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--image-dir", type=Path, dest="image_dir")
    parser.add_argument("--cache-dir", type=Path, dest="cache_dir",
                        default=Path("data/cache"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--prob-cache", type=Path, dest="prob_cache",
                        default=Path("runs/prob_cache"))
    parser.add_argument("--limit", type=int, default=60,
                        help="held-out observations to tune on")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tile-size", type=int, default=512, dest="tile_size")
    parser.add_argument("--tta", action="store_true")
    parser.add_argument("--metric", type=str, default="matched_dice",
                        help="metric to maximise")
    parser.add_argument("--out", type=Path)

    parser.add_argument("--thresholds", type=float, nargs="+",
                        default=[0.5, 0.7, 0.85, 0.93])
    parser.add_argument("--min-confidence", type=float, nargs="+",
                        dest="min_confidence", default=[0.0, 0.6, 0.75, 0.85])
    parser.add_argument("--merge-gap", type=float, nargs="+", dest="merge_gap",
                        default=[18.0, 40.0, 72.0])
    parser.add_argument("--min-area-fraction", type=float, nargs="+",
                        dest="min_area_fraction", default=[1.2e-4, 4e-4, 1e-3])
    args = parser.parse_args()

    if args.data_dir and not args.image_dir:
        args.image_dir = discover(args.data_dir).train_dir
    annotations = resolve_annotations(args.annotations, args.image_dir, args.data_dir)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = MagfiloDataset(annotations, args.image_dir, args.cache_dir)

    # Tune on the tail of the set, which the default split holds out.
    total = len(dataset)
    n = min(total, args.limit)
    indices = list(range(total - n, total))
    print(f"tuning on {n} held-out observations of {total}\n")

    paths = cache_probabilities(
        args.checkpoint, dataset, indices, args.prob_cache,
        args.device, args.tile_size, args.tta,
    )

    grid = list(itertools.product(
        args.thresholds, args.min_confidence, args.merge_gap, args.min_area_fraction
    ))
    print(f"\nsweeping {len(grid)} configurations over the cached maps")
    print(f"{'thr':>6}{'conf':>6}{'gap':>7}{'minarea':>9}"
          f"{'matchedD':>10}{'fgDice':>8}{'PQ':>7}{'RQ':>7}"
          f"{'inst':>7}{'spur':>7}{'1->m':>6}{'miss':>6}")

    results: list[dict] = []
    for threshold, confidence, gap, area in grid:
        config = InstanceConfig(
            threshold=threshold,
            min_confidence=confidence,
            merge_gap=gap,
            min_area_fraction=area,
        )
        summary = score(dataset, indices, paths, config)
        results.append({
            "threshold": threshold, "min_confidence": confidence,
            "merge_gap": gap, "min_area_fraction": area, **summary,
        })
        print(f"{threshold:>6.2f}{confidence:>6.2f}{gap:>7.1f}{area:>9.1e}"
              f"{summary['matched_dice']:>10.4f}{summary['foreground_dice']:>8.4f}"
              f"{summary['pq']:>7.4f}{summary['rq']:>7.4f}"
              f"{summary['n_instances']:>7.1f}{summary['spurious']:>7.0f}"
              f"{summary['one_to_many']:>6.0f}{summary['missed']:>6.0f}", flush=True)

    best = max(results, key=lambda r: r.get(args.metric, 0.0))
    print("\n" + "=" * 74)
    print(f"BEST by {args.metric}")
    print("=" * 74)
    for key in ("threshold", "min_confidence", "merge_gap", "min_area_fraction",
                "matched_dice", "matched_dice_over_truth", "mean_paired_dice",
                "foreground_dice", "pq", "rq", "n_instances", "spurious",
                "one_to_many", "missed"):
        if key in best:
            print(f"  {key:24s} {best[key]:.4f}")

    print("\nUse it with:")
    print(f"  python scripts/predict.py --images data/test \\\n"
          f"      --checkpoint {args.checkpoint} --out submission.csv \\\n"
          f"      --threshold {best['threshold']} "
          f"--min-confidence {best['min_confidence']} \\\n"
          f"      --merge-gap {best['merge_gap']} "
          f"--min-area-fraction {best['min_area_fraction']}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump({"best": best, "all": results}, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
