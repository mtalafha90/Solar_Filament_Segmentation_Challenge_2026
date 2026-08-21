#!/usr/bin/env python3
"""Segment filaments in a directory of full-disk images and write a submission.

Writes the competition's CSV by default -- one row per predicted filament, keyed
``<image_id>_<n>``, with the mask as pycocotools RLE counts::

    python scripts/predict.py --images data/test \
        --checkpoint runs/filanet/best.pt --out submission.csv

Use ``--classical`` in place of ``--checkpoint`` for the training-free detector,
and ``--format coco`` or ``--format png`` for the other writers.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from filaseg.classical import detect
from filaseg.data.coco import normalise_id
from filaseg.data.io import IMAGE_SUFFIXES, FITS_SUFFIXES, read_image
from filaseg.inference import InferenceConfig, predict
from filaseg.postprocess.instances import InstanceConfig
from filaseg.submission import (
    summarise_predictions,
    write_challenge_csv,
    write_coco,
    write_label_pngs,
    write_rle_csv,
)


def gather_images(directory: Path) -> list[Path]:
    suffixes = FITS_SUFFIXES | IMAGE_SUFFIXES | {".npy"}
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in suffixes)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--classical", action="store_true",
                        help="use the training-free detector instead of a network")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--format",
                        choices=("challenge", "coco", "csv", "png"),
                        default="challenge",
                        help="'challenge' writes the competition CSV: one row per "
                             "filament, pycocotools RLE counts (the default)")
    parser.add_argument("--threshold", type=float,
                        help="override the threshold stored in the checkpoint")
    parser.add_argument("--tile-size", type=int, default=512, dest="tile_size")
    parser.add_argument("--no-tta", dest="tta", action="store_false")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--min-area", type=int, default=40, dest="min_area")
    parser.add_argument("--min-confidence", type=float, default=0.0,
                        dest="min_confidence",
                        help="drop instances whose mean probability is below this; "
                             "tune with scripts/tune_postprocess.py")
    parser.add_argument("--min-area-fraction", type=float, default=1.2e-4,
                        dest="min_area_fraction")
    parser.add_argument("--merge-gap", type=float, default=18.0, dest="merge_gap")
    parser.add_argument("--save-labels", type=Path, dest="save_labels",
                        help="also write label maps as PNGs to this directory")
    args = parser.parse_args()

    if not args.classical and args.checkpoint is None:
        raise SystemExit("give --checkpoint, or --classical for the training-free detector")

    paths = gather_images(args.images)
    if not paths:
        raise SystemExit(f"no images found under {args.images}")
    print(f"found {len(paths)} images")

    model = None
    threshold = args.threshold if args.threshold is not None else 0.5
    if not args.classical:
        from filaseg.train import load_model

        model, stored_threshold = load_model(args.checkpoint, args.device)
        if args.threshold is None:
            threshold = stored_threshold
        print(f"loaded {args.checkpoint} (threshold {threshold:.2f})")

    instance_config = InstanceConfig(
        threshold=threshold,
        min_area=args.min_area,
        min_confidence=args.min_confidence,
        min_area_fraction=args.min_area_fraction,
        merge_gap=args.merge_gap,
    )
    inference_config = InferenceConfig(
        tile_size=args.tile_size, tta=args.tta, device=args.device
    )

    records: list[tuple] = []
    all_labels: list[np.ndarray] = []
    started = time.time()

    for index, path in enumerate(paths, start=1):
        image = read_image(path)
        if args.classical:
            labels = detect(image)
            probability = None
        else:
            from filaseg.postprocess.instances import extract_instances

            probability, valid, _ = predict(model, image, inference_config)
            labels = extract_instances(probability, valid, instance_config)

        records.append((path.stem, labels, probability))
        all_labels.append(labels)
        if index % 10 == 0 or index == len(paths):
            print(f"  {index}/{len(paths)}  ({time.time() - started:.0f}s)", flush=True)

    if args.format == "challenge":
        count = write_challenge_csv(
            args.out, [(name, labels) for name, labels, _ in records]
        )
        print(f"wrote {count} filament rows to {args.out}")
    elif args.format == "coco":
        # The image id is the file stem, which is what the annotations use.
        # MAGFiLO keys observations by the original GONG frame name, so deriving
        # an integer from it would produce ids the grader cannot match.
        coco_records = [
            (normalise_id(name), labels, probability)
            for name, labels, probability in records
        ]
        count = write_coco(args.out, coco_records)
        print(f"wrote {count} instances to {args.out}")
    elif args.format == "csv":
        count = write_rle_csv(args.out, records)
        print(f"wrote {count} rows to {args.out}")
    else:
        count = write_label_pngs(args.out, [(n, l) for n, l, _ in records])
        print(f"wrote {count} label maps to {args.out}")

    if args.save_labels:
        write_label_pngs(args.save_labels, [(n, l) for n, l, _ in records])

    summary = summarise_predictions(all_labels)
    print("\nsubmission summary:")
    print(json.dumps({k: round(float(v), 4) for k, v in summary.items()}, indent=2))


if __name__ == "__main__":
    main()
