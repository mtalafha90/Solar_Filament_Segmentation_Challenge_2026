#!/usr/bin/env python3
"""Score predictions against ground-truth annotations.

Reports what the challenge ranks on -- Panoptic Quality and the mean Dice score
-- alongside the fragmentation and over-merging counts it penalises, the pixel
metrics, and the end-to-end time per frame, which is also assessed.

Example::

    python scripts/evaluate.py --annotations data/magfilo/annotations.json \
        --image-dir data/magfilo/images --checkpoint runs/filanet/best.pt
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from filaseg.classical import detect
from filaseg.data.dataset import MagfiloDataset
from filaseg.data.layout import discover, resolve_annotations
from filaseg.metrics import aggregate, evaluate
from filaseg.postprocess.instances import InstanceConfig, extract_instances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, dest="data_dir",
                        help="dataset root; annotations and images found inside")
    parser.add_argument("--annotations", type=Path,
                        help="annotation JSON; discovered automatically if omitted")
    parser.add_argument("--image-dir", type=Path, dest="image_dir")
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

    if args.data_dir and not args.image_dir:
        args.image_dir = discover(args.data_dir).train_dir
    if args.image_dir is None:
        raise SystemExit("--image-dir is required (or pass --data-dir)")
    try:
        args.annotations = resolve_annotations(
            args.annotations, args.image_dir, args.data_dir
        )
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

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
    started = time.time()
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

    elapsed = time.time() - started
    summary = aggregate([{k: v for k, v in r.items() if k != "image_id"} for r in per_image])
    summary["seconds_per_image"] = elapsed / max(len(per_image), 1)

    groups = [
        ("RANKED ON", ["pq", "dice"]),
        ("PANOPTIC BREAKDOWN", ["sq", "rq", "pq_tp", "pq_fp", "pq_fn"]),
        ("FRAGMENTATION AND OVER-MERGING",
         ["one_to_one", "one_to_many", "many_to_one", "missed", "spurious",
          "fragments_per_split"]),
        ("PIXEL OVERLAP",
         ["iou", "precision", "recall", "f1", "cl_dice", "msiou"]),
        ("DETECTION",
         ["hit_rate", "miss_rate", "false_discovery_rate", "mean_pairwise_iou",
          "AP@0.25", "AP@0.50", "AP@0.75", "mAP"]),
        ("COUNTS", ["n_predicted", "n_truth", "n_matched", "n_images"]),
        ("EFFICIENCY", ["seconds_per_image"]),
    ]
    print("\n" + "=" * 58)
    print("RESULTS")
    print("=" * 58)
    for title, keys in groups:
        present = [k for k in keys if k in summary]
        if not present:
            continue
        print(f"\n  {title}")
        for key in present:
            print(f"    {key:24s} {summary[key]:.4f}")

    # Distributions, which the challenge asks for explicitly.
    if per_image:
        print("\n  DISTRIBUTIONS across observations")
        for key in ("dice", "iou", "pq"):
            values = np.array([r[key] for r in per_image if key in r])
            if values.size:
                quartiles = np.percentile(values, [25, 50, 75])
                print(f"    {key:8s} min {values.min():.3f}  "
                      f"q1 {quartiles[0]:.3f}  median {quartiles[1]:.3f}  "
                      f"q3 {quartiles[2]:.3f}  max {values.max():.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump({"summary": summary, "per_image": per_image}, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
