#!/usr/bin/env python3
"""Grid-search the classical detector's thresholds on a labelled set.

The defaults shipped in :class:`~filaseg.classical.ClassicalConfig` were chosen
on synthetic data.  Run this once on real annotated observations to re-tune
them, then pass the winning values through ``--config``.

Example::

    python scripts/tune_classical.py --annotations data/magfilo/annotations.json \
        --image-dir data/magfilo/images --limit 40
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from filaseg.classical import ClassicalConfig, choose_thresholds, hysteresis, score_map
from filaseg.data.dataset import MagfiloDataset
from filaseg.metrics import aggregate, evaluate
from filaseg.postprocess.instances import InstanceConfig, extract_instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True, dest="image_dir")
    parser.add_argument("--cache-dir", type=Path, dest="cache_dir")
    parser.add_argument("--limit", type=int, default=30)
    parser.add_argument("--ridge-weights", type=float, nargs="+",
                        default=[0.0, 0.3, 0.5, 0.7], dest="ridge_weights")
    parser.add_argument("--coverage", type=float, nargs="+",
                        default=[0.006, 0.010, 0.014, 0.020],
                        help="candidate values for expected_coverage")
    parser.add_argument("--growth", type=float, nargs="+",
                        default=[2.0, 3.0, 4.0, 6.0],
                        help="candidate values for growth_factor")
    parser.add_argument("--metric", type=str, default="iou",
                        help="metric to maximise (iou, msiou, cl_dice, mean_pairwise_iou)")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    dataset = MagfiloDataset(args.annotations, args.image_dir, args.cache_dir)
    n = min(len(dataset), args.limit)
    prepared = [dataset[i] for i in range(n)]
    print(f"tuning on {n} observations, maximising {args.metric}")

    coverage = float(np.mean([p.mask.sum() / max(p.valid.sum(), 1) for p in prepared]))
    print(f"filaments cover {100 * coverage:.3f}% of the disk on average "
          f"-> expected_coverage should sit somewhat below that, since it "
          f"describes filament cores before hysteresis grows them")

    results: list[dict] = []
    for weight in args.ridge_weights:
        config = ClassicalConfig(ridge_weight=weight)
        scores = [score_map(p.image, p.valid, config) for p in prepared]
        for expected, growth in itertools.product(args.coverage, args.growth):
            trial = ClassicalConfig(ridge_weight=weight, expected_coverage=expected,
                                    growth_factor=growth)
            rows = []
            for p, score in zip(prepared, scores):
                low, high = choose_thresholds(score[p.valid], trial)
                mask = hysteresis(score, low, high, p.valid)
                labels = extract_instances(mask.astype(np.float32), p.valid,
                                           InstanceConfig(min_area=trial.instance.min_area))
                rows.append(evaluate(labels, p.instances, p.valid))
            summary = aggregate(rows)
            results.append({"ridge_weight": weight, "expected_coverage": expected,
                            "growth_factor": growth, **summary})
            print(f"  w={weight:.2f} coverage={expected:.3f} growth={growth:.1f} -> "
                  f"IoU {summary['iou']:.4f}  MSIoU {summary['msiou']:.4f}  "
                  f"clDice {summary['cl_dice']:.4f}  hit {summary['hit_rate']:.3f}",
                  flush=True)

    best = max(results, key=lambda r: r.get(args.metric, 0.0))
    print("\nbest configuration:")
    print(json.dumps({k: round(float(v), 4) for k, v in best.items()
                      if k in ("ridge_weight", "expected_coverage", "growth_factor",
                               "iou", "msiou", "cl_dice", "hit_rate")}, indent=2))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
