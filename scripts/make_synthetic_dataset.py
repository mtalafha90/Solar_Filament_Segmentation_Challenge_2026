#!/usr/bin/env python3
"""Generate a synthetic dataset laid out exactly like MAGFiLO.

This exists so the pipeline can be developed, tested and benchmarked before the
competition data has been downloaded, and so the automated tests have something
realistic to run against.  The output directory mirrors the real layout::

    <out>/images/synth_00000.npy
    <out>/annotations.json

Run ``python scripts/make_synthetic_dataset.py --help`` for the options.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from filaseg.data.coco import mask_to_polygons  # noqa: E402
from filaseg.data.synthetic import generate_observation  # noqa: E402


def build(
    out_dir: Path,
    n_images: int,
    size: int,
    n_filaments: int,
    n_sunspots: int,
    seed: int,
) -> dict:
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)

    images: list[dict] = []
    annotations: list[dict] = []
    annotation_id = 1

    for index in range(n_images):
        observation = generate_observation(
            size=size,
            n_filaments=n_filaments,
            n_sunspots=n_sunspots,
            seed=seed + index,
        )
        name = f"synth_{index:05d}.npy"
        np.save(image_dir / name, observation.image.astype(np.float32))

        images.append(
            {
                "id": index + 1,
                "file_name": name,
                "height": size,
                "width": size,
                # Extra fields mirroring what MAGFiLO records per observation.
                "solar_radius": round(observation.radius, 3),
                "centre_y": round(observation.centre_y, 3),
                "centre_x": round(observation.centre_x, 3),
            }
        )

        for filament in observation.filaments:
            rows, cols = np.nonzero(filament.mask)
            if rows.size == 0:
                continue
            polygons = mask_to_polygons(filament.mask, tolerance=0.6)
            if not polygons:
                continue
            # Subsample the spine so the JSON stays a sensible size.
            spine = filament.spine[:: max(1, len(filament.spine) // 60)]
            annotations.append(
                {
                    "id": annotation_id,
                    "image_id": index + 1,
                    "category_id": 1,
                    "iscrowd": 0,
                    "area": float(filament.mask.sum()),
                    "bbox": [
                        float(cols.min()),
                        float(rows.min()),
                        float(cols.max() - cols.min() + 1),
                        float(rows.max() - rows.min() + 1),
                    ],
                    "segmentation": polygons,
                    # MAGFiLO stores spines as (x, y); match that convention.
                    "spine": [
                        [round(float(x), 2), round(float(y), 2)] for y, x in spine
                    ],
                    "chirality": int(filament.chirality),
                }
            )
            annotation_id += 1

    return {
        "info": {
            "description": "Synthetic MAGFiLO-like dataset for pipeline testing",
            "version": "1.0",
            "note": "Generated data. Not real solar observations.",
        },
        "licenses": [],
        "categories": [{"id": 1, "name": "filament", "supercategory": "solar_feature"}],
        "images": images,
        "annotations": annotations,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("data/synthetic"))
    parser.add_argument("--n-images", type=int, default=24)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--n-filaments", type=int, default=8)
    parser.add_argument("--n-sunspots", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    coco = build(
        args.out, args.n_images, args.size, args.n_filaments, args.n_sunspots, args.seed
    )
    args.out.mkdir(parents=True, exist_ok=True)
    with (args.out / "annotations.json").open("w", encoding="utf-8") as handle:
        json.dump(coco, handle)

    print(f"wrote {len(coco['images'])} images and {len(coco['annotations'])} filaments")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
