#!/usr/bin/env python3
"""Check a real dataset before training on it, and report what is in it.

Run this first. It verifies that the annotations parse, that every referenced
image can be found and read, and that images and annotations agree on size; then
it reports the statistics that determine how the detectors should be configured
-- above all the fraction of the solar disk covered by filaments, which sets the
classical detector's coverage prior and the network's positive-class weight.

Example::

    python scripts/inspect_data.py --data-dir data
    python scripts/inspect_data.py --annotations data/train/annotations.json \
        --image-dir data/train --test-dir data/test
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from filaseg.data.coco import CHIRALITY_NAMES, load_coco, rescale_record, summarise
from filaseg.data.io import FITS_SUFFIXES, IMAGE_SUFFIXES, find_image, read_image
from filaseg.preprocessing.photometry import preprocess

SUFFIXES = FITS_SUFFIXES | IMAGE_SUFFIXES | {".npy"}


def find_layout(data_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Work out where the annotations, training images and test images live.

    Handles the usual competition layout of ``data/train`` plus ``data/test``,
    with the annotation JSON anywhere inside the training folder.
    """
    train_dir = next(
        (d for d in (data_dir / "train", data_dir / "training", data_dir) if d.is_dir()),
        None,
    )
    test_dir = next(
        (d for d in (data_dir / "test", data_dir / "testing") if d.is_dir()), None
    )

    annotations = None
    for candidate_dir in [d for d in (train_dir, data_dir) if d is not None]:
        candidates = sorted(candidate_dir.glob("*.json")) + sorted(
            candidate_dir.glob("*/*.json")
        )
        if candidates:
            # Prefer the largest, which is almost always the real annotation file.
            annotations = max(candidates, key=lambda p: p.stat().st_size)
            break
    return annotations, train_dir, test_dir


def count_images(directory: Path | None) -> list[Path]:
    if directory is None or not directory.is_dir():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in SUFFIXES)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"),
                        dest="data_dir", help="folder holding train/ and test/")
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--image-dir", type=Path, dest="image_dir")
    parser.add_argument("--test-dir", type=Path, dest="test_dir")
    parser.add_argument("--sample", type=int, default=12,
                        help="how many observations to preprocess for statistics")
    parser.add_argument("--out", type=Path, help="write the report as JSON")
    args = parser.parse_args()

    annotations, image_dir, test_dir = find_layout(args.data_dir)
    annotations = args.annotations or annotations
    image_dir = args.image_dir or image_dir
    test_dir = args.test_dir or test_dir

    print("=" * 66)
    print("LAYOUT")
    print("=" * 66)
    print(f"  annotations : {annotations}")
    print(f"  train images: {image_dir}")
    print(f"  test images : {test_dir}")

    train_images = count_images(image_dir)
    test_images = count_images(test_dir)
    print(f"\n  {len(train_images)} training images, {len(test_images)} test images")
    if train_images:
        extensions = Counter(p.suffix.lower() for p in train_images)
        print(f"  formats: {dict(extensions)}")

    if annotations is None or not Path(annotations).exists():
        raise SystemExit(
            "\nNo annotation JSON found. Pass --annotations explicitly."
        )

    records, meta = load_coco(annotations)
    stats = summarise(records)

    print("\n" + "=" * 66)
    print("ANNOTATIONS")
    print("=" * 66)
    print(f"  categories: {meta.get('categories')}")
    for key, value in stats.items():
        if key == "chirality":
            continue
        print(f"  {key:28s} {value}")
    print(f"  chirality: {stats['chirality']}")

    # Check every referenced image resolves, and that sizes agree.
    print("\n" + "=" * 66)
    print("INTEGRITY")
    print("=" * 66)
    missing: list[str] = []
    mismatched: list[tuple[str, tuple, tuple]] = []
    shapes: Counter = Counter()

    for record in records:
        try:
            path = find_image(image_dir, record.file_name)
        except FileNotFoundError:
            missing.append(record.file_name)
            continue
        if len(shapes) < args.sample or record is records[-1]:
            try:
                image = read_image(path)
            except Exception as error:  # noqa: BLE001 - report, do not crash
                missing.append(f"{record.file_name} (unreadable: {error})")
                continue
            shapes[image.shape] += 1
            if (record.height, record.width) not in ((0, 0), image.shape):
                mismatched.append(
                    (record.file_name, (record.height, record.width), image.shape)
                )

    print(f"  images referenced but not found: {len(missing)}")
    for name in missing[:5]:
        print(f"    - {name}")
    print(f"  image shapes seen: {dict(shapes)}")
    if mismatched:
        print(f"\n  WARNING: {len(mismatched)} size mismatch(es) between annotations "
              f"and images.")
        for name, annotated, actual in mismatched[:3]:
            print(f"    - {name}: annotation {annotated} vs image {actual}")
        print("  The loader rescales annotations to the image automatically, but "
              "check\n  that this is what the organisers intended.")
    else:
        print("  annotation and image sizes agree")

    # Preprocess a sample to measure the statistics that drive configuration.
    print("\n" + "=" * 66)
    print("MEASUREMENTS (from a sample of observations)")
    print("=" * 66)
    coverage: list[float] = []
    radii: list[float] = []
    counts: list[int] = []
    sample = records[: max(1, args.sample)]

    for record in sample:
        try:
            image = read_image(find_image(image_dir, record.file_name))
        except FileNotFoundError:
            continue
        if (record.height, record.width) != image.shape:
            rescale_record(record, image.shape[0], image.shape[1])
        _, valid, disk = preprocess(image)
        mask = record.semantic_mask() & valid
        coverage.append(float(mask.sum() / max(valid.sum(), 1)))
        radii.append(disk.radius)
        counts.append(len(record.annotations))

    report: dict = {
        "annotations": str(annotations),
        "image_dir": str(image_dir),
        "test_dir": str(test_dir) if test_dir else None,
        "n_train_images": len(train_images),
        "n_test_images": len(test_images),
        "image_shapes": {str(k): v for k, v in shapes.items()},
        "n_missing_images": len(missing),
        "n_size_mismatches": len(mismatched),
        **{k: v for k, v in stats.items() if k != "chirality"},
        "chirality": stats["chirality"],
    }

    if coverage:
        mean_coverage = float(np.mean(coverage))
        report.update(
            {
                "disk_coverage_mean": mean_coverage,
                "disk_coverage_min": float(np.min(coverage)),
                "disk_coverage_max": float(np.max(coverage)),
                "solar_radius_mean_px": float(np.mean(radii)),
                "filaments_per_image_sampled": float(np.mean(counts)),
            }
        )
        print(f"  solar radius            {np.mean(radii):.1f} px")
        print(f"  filaments per image     {np.mean(counts):.1f}")
        print(f"  disk coverage by filaments:")
        print(f"      mean {100 * mean_coverage:.3f}%   "
              f"min {100 * np.min(coverage):.3f}%   "
              f"max {100 * np.max(coverage):.3f}%")

        print("\n" + "=" * 66)
        print("SUGGESTED SETTINGS")
        print("=" * 66)
        # The classical detector seeds filament *cores*, a fraction of the full
        # annotated extent, and hysteresis grows out to the rest. The right
        # fraction is dataset-specific: raising it buys recall and costs
        # precision, and which way that moves IoU cannot be predicted from
        # coverage alone. So suggest a range to search, not a single value.
        low = max(0.002, mean_coverage * 0.25)
        high = max(0.004, mean_coverage * 0.8)
        centre = max(0.003, mean_coverage * 0.4)
        print(f"  classical detector : expected_coverage near {centre:.4f}")
        print(f"                       search {low:.4f} to {high:.4f} -- this knob")
        print(f"                       trades recall against precision, so tune it:")
        print(f"    python scripts/tune_classical.py --annotations {annotations} \\")
        print(f"        --image-dir {image_dir} \\")
        print(f"        --coverage {low:.4f} {centre:.4f} {high:.4f}")
        # A pos_weight near sqrt(1/coverage) is a reasonable starting point.
        pos_weight = float(np.clip(np.sqrt(1.0 / max(mean_coverage, 1e-4)), 2.0, 20.0))
        print(f"  network            : pos_weight = {pos_weight:.1f}")
        radius = float(np.mean(radii))
        tile = 512 if radius > 700 else 256
        print(f"  network            : patch_size = {tile}, val_tile = {tile}")
        report["suggested_expected_coverage"] = centre
        report["suggested_coverage_range"] = [low, high]
        report["suggested_pos_weight"] = pos_weight
        report["suggested_patch_size"] = tile

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
