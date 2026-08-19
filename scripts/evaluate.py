#!/usr/bin/env python3
"""Score predictions against ground-truth annotations.

Reports every metric the challenge uses: pixel IoU, precision, recall, clDice,
multi-scale IoU, hit and miss rates, and AP at several IoU thresholds.

Example::

    python scripts/evaluate.py --annotations data/magfilo/annotations.json \
        --image-dir data/magfilo/images --checkpoint runs/filanet/best.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from filaseg.classical import detect
from filaseg.data.dataset import MagfiloDataset
from filaseg.metrics import aggregate, evaluate
from filaseg.postprocess.instances import InstanceConfig, extract_instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True, dest="image_dir")
    parser.add_argument("--cache-dir", type=Path, dest="cache_dir")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--classical", action="store_true")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--limit", type=int, help="score only the first N observations")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--tile-size", type=int, default=512, dest="tile_size")
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.add_argument("--min-area", type=int, default=40, dest="min_area")
    parser.add_argument("--out", type=Path, help="write per-image metrics as JSON")
    args = parser.parse_args()

    if not args.classical and args.checkpoint is None:
        raise SystemExit("give --checkpoint, or --classical for the training-free detector")

    dataset = MagfiloDataset(args.annotations, args.image_dir, args.cache_dir)
    indices = range(min(len(dataset), args.limit) if args.limit else len(dataset))
    print(f"scoring {len(list(indices))} observations")

    model = None
    threshold = args.threshold if args.threshold is not None else 0.5
    if not args.classical:
        from filaseg.train import load_model

        model, stored = load_model(args.checkpoint, args.device)
        if args.threshold is None:
            threshold = stored
        print(f"loaded {args.checkpoint} (threshold {threshold:.2f})")

    from filaseg.data.io import find_image, read_image
    from filaseg.inference import InferenceConfig, predict_probability

    inference_config = InferenceConfig(
        tile_size=args.tile_size, tta=args.tta, device=args.device
    )
    instance_config = InstanceConfig(threshold=threshold, min_area=args.min_area)

    per_image: list[dict] = []
    for position, index in enumerate(indices, start=1):
        prepared = dataset[index]
        if args.classical:
            raw = read_image(find_image(args.image_dir, prepared.file_name))
            labels = detect(raw)
        else:
            probability = predict_probability(model, prepared.input_stack(), inference_config)
            labels = extract_instances(probability * prepared.valid, prepared.valid,
                                       instance_config)

        # Ground-truth instances come straight from the annotations.
        scores = evaluate(labels, prepared.instances, prepared.valid)
        scores["image_id"] = prepared.image_id
        per_image.append(scores)
        if position % 10 == 0 or position == len(list(indices)):
            print(f"  {position} done", flush=True)

    summary = aggregate([{k: v for k, v in r.items() if k != "image_id"} for r in per_image])
    print("\n" + "=" * 58)
    print("RESULTS")
    print("=" * 58)
    order = ["iou", "dice", "precision", "recall", "f1", "cl_dice", "msiou",
             "hit_rate", "miss_rate", "false_discovery_rate", "mean_pairwise_iou",
             "AP@0.25", "AP@0.50", "AP@0.75", "mAP",
             "n_predicted", "n_truth", "n_matched", "n_images"]
    for key in order:
        if key in summary:
            print(f"  {key:22s} {summary[key]:.4f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "per_image": per_image}, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
