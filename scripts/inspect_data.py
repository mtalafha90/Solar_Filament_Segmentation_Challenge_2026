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
import warnings
from collections import Counter
from pathlib import Path

import _bootstrap  # noqa: F401

import numpy as np

from filaseg.data.coco import load_coco, summarise
from filaseg.data.dataset import MagfiloDataset
from filaseg.data.io import read_image, resolve_images
from filaseg.data.layout import count_images, discover, resolve_annotations


def find_layout(data_dir: Path) -> tuple[Path | None, Path | None, Path | None]:
    """Locate the annotations, training images and test images."""
    layout = discover(data_dir)
    return layout.annotations, layout.train_dir, layout.test_dir


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

    try:
        annotations = resolve_annotations(annotations, image_dir, args.data_dir)
    except FileNotFoundError as error:
        raise SystemExit(f"\n{error}") from error

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

    # Check that every referenced image resolves, that no two records claim the
    # same file, and that images and annotations agree on size.
    print("\n" + "=" * 66)
    print("INTEGRITY")
    print("=" * 66)

    resolved, missing, collisions = resolve_images(
        image_dir, [record.file_name for record in records]
    )

    distinct_names = {record.file_name for record in records}
    print(f"  annotation records           {len(records)}")
    print(f"  distinct file names in them  {len(distinct_names)}")
    print(f"  image files on disk          {len(train_images)}")
    print(f"  file names resolved          {len(resolved)}")
    print(f"  file names with no image     {len(missing)}")
    for name in missing[:3]:
        print(f"    - {name}")
    if len(missing) > 3:
        print(f"    ... and {len(missing) - 3} more")

    if len(records) > len(distinct_names):
        print(f"\n  {len(records) - len(distinct_names)} record(s) repeat a frame "
              f"already listed. Their filaments are")
        print("  merged on load, so no filament is presented as background.")

    if missing and resolved:
        print("\n  Records with no image are normal: the annotation file covers")
        print("  more observations than this split contains. They are skipped.")

    # If almost nothing resolved, the names on disk differ from the names in the
    # annotations. Show both, so the mismatch is obvious rather than a mystery.
    if distinct_names and len(resolved) < 0.5 * len(distinct_names):
        print("\n  MOST FILE NAMES DID NOT RESOLVE. Compare them:")
        print("    annotations say : "
              + ", ".join(sorted(distinct_names)[:3]))
        print("    on disk         : "
              + ", ".join(p.name for p in train_images[:3]))
        directories = sorted({p.parent for p in train_images})[:3]
        print("    images found in : "
              + ", ".join(str(d) for d in directories))
        print("  Point --image-dir at the directory holding the images.")

    if collisions:
        print(f"\n  ERROR: {len(collisions)} image file(s) are claimed by more than")
        print("  one record. Annotations would be paired with the wrong frame.")
        for path, names in list(collisions.items())[:3]:
            print(f"    - {path.name} <- {', '.join(names[:3])}")
    else:
        print("  no two records resolve to the same file")

    mismatched: list[tuple[str, tuple, tuple]] = []
    shapes: Counter = Counter()
    unreadable: list[str] = []
    sampled = 0
    seen_paths: set[Path] = set()
    for record in records:
        if sampled >= args.sample:
            break
        path = resolved.get(record.file_name)
        if path is None or path in seen_paths:
            continue
        seen_paths.add(path)
        sampled += 1
        try:
            image = read_image(path)
        except Exception as error:  # noqa: BLE001 - report, do not crash
            unreadable.append(f"{record.file_name} ({error})")
            continue
        shapes[image.shape] += 1
        if (record.height, record.width) not in ((0, 0), image.shape):
            mismatched.append(
                (record.file_name, (record.height, record.width), image.shape)
            )

    print(f"\n  image shapes, from {sampled} sampled: {dict(shapes)}")
    if unreadable:
        print(f"  unreadable images: {len(unreadable)}")
        for name in unreadable[:3]:
            print(f"    - {name}")
    if mismatched:
        print(f"\n  WARNING: {len(mismatched)} size mismatch(es) between annotations "
              f"and images.")
        for name, annotated, actual in mismatched[:3]:
            print(f"    - {name}: annotation {annotated} vs image {actual}")
        print("  The loader rescales annotations to the image automatically, but "
              "check\n  that this is what the organisers intended.")
    else:
        print("  annotation and image sizes agree")

    # Measure through the dataset itself, so the numbers describe exactly what
    # training will see: duplicate frames merged, records without images
    # dropped, annotations clipped to the disk.
    print("\n" + "=" * 66)
    print("MEASUREMENTS (as the training pipeline will see them)")
    print("=" * 66)

    coverage: list[float] = []
    radii: list[float] = []
    counts: list[int] = []
    n_observations = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            dataset = MagfiloDataset(annotations, image_dir)
        except ValueError as error:
            raise SystemExit(f"\n{error}") from error

    n_observations = len(dataset)
    print(f"  usable observations     {n_observations}")
    if n_observations == 0:
        raise SystemExit(
            "\nNo observation could be loaded. Check that --image-dir points at "
            "the images these annotations describe."
        )

    for index in range(min(n_observations, max(1, args.sample))):
        prepared = dataset[index]
        coverage.append(float(prepared.mask.sum() / max(prepared.valid.sum(), 1)))
        radii.append(prepared.disk.radius)
        counts.append(int(prepared.instances.max()))

    report: dict = {
        "n_records": len(records),
        "n_distinct_file_names": len(distinct_names),
        "n_records_resolved": len(resolved),
        "n_records_without_image": len(missing),
        "n_collisions": len(collisions),
        "annotations": str(annotations),
        "image_dir": str(image_dir),
        "test_dir": str(test_dir) if test_dir else None,
        "n_train_images": len(train_images),
        "n_test_images": len(test_images),
        "image_shapes": {str(k): v for k, v in shapes.items()},
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
        disk_area = np.pi * radius**2
        print(f"  post-processing    : min_area resolves to "
              f"{max(40, int(round(1.2e-4 * disk_area)))} px at this resolution")
        report["suggested_min_area"] = max(40, int(round(1.2e-4 * disk_area)))
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
