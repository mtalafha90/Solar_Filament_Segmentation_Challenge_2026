#!/usr/bin/env python3
"""Tune instance construction on a fixed checkpoint, without retraining.

The leaderboard matches predictions to ground truth one filament at a time, so
a model can cover the right pixels and still score badly by splitting filaments
apart or inventing extra ones. Neural inference is the expensive part and does
not change while post-processing is tuned, so probability maps are computed
once and cached and every geometry sweep reuses them.

This script reconstructs the exact grouped validation split stored in the
checkpoint configuration. It never assumes that the tail of the annotation file
is held out. When ``--limit`` is used, the requested number of validation
annotation records is sampled evenly from that true held-out split.

Probability maps are cached per physical image rather than per tuning subset.
That means a 60-record tuning run can be expanded to the full validation split
without recomputing images that were already inferred, including duplicate
annotation records for the same observation.

Example::

    python scripts/tune_postprocess.py --data-dir data \
        --checkpoint runs/cpu_filanet_20epoch/best.pt --limit 60
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import shutil
import time
import warnings
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
from filaseg.train import split_ids


def _checkpoint_blob(checkpoint: Path) -> dict:
    """Load a checkpoint on CPU so its split and inference settings can be reused."""
    import torch

    return torch.load(checkpoint, map_location="cpu", weights_only=False)


def _file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Content hash used to prevent probability-cache collisions between models."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _evenly_spaced(indices: list[int], limit: int) -> list[int]:
    """Take a deterministic, evenly spaced subset without changing its order."""
    if limit <= 0 or len(indices) <= limit:
        return indices
    step = len(indices) / limit
    return [indices[int(i * step)] for i in range(limit)]


def validation_dataset(
    annotations: Path,
    image_dir: Path,
    cache_dir: Path,
    checkpoint_config: dict,
    limit: int,
) -> tuple[MagfiloDataset, list[int], dict[str, float | int]]:
    """Rebuild the exact grouped validation split that training used."""
    source = MagfiloDataset(annotations, image_dir, cache_dir)
    val_fraction = float(checkpoint_config.get("val_fraction", 0.15))
    seed = int(checkpoint_config.get("seed", 0))
    _, val_indices = split_ids(
        len(source), val_fraction, seed, groups=source.group_keys
    )

    val_source = MagfiloDataset(
        annotations,
        image_dir,
        cache_dir,
        image_ids=[source.records[i].image_id for i in val_indices],
    )
    indices = _evenly_spaced(list(range(len(val_source))), limit)
    split_info: dict[str, float | int] = {
        "seed": seed,
        "val_fraction": val_fraction,
        "full_validation_records": len(val_source),
        "full_validation_groups": len(set(val_source.group_keys)),
        "selected_records": len(indices),
        "selected_groups": len({val_source.group_keys[i] for i in indices}),
    }
    return val_source, indices, split_info


def _record_cache_key(dataset: MagfiloDataset, index: int) -> str:
    """Stable physical-image key; duplicate annotator records intentionally share it."""
    return str(dataset.group_keys[index])


def _record_cache_name(key: str) -> str:
    """Filesystem-safe deterministic name for one physical image."""
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return f"{digest}.npy"


def probability_cache_namespace(
    checkpoint: Path,
    dataset: MagfiloDataset,
    indices: list[int],
    tile_size: int,
    tta: bool,
) -> tuple[str, dict]:
    """Return the cache namespace for one inference configuration.

    The namespace depends on the model and inference settings, but deliberately
    not on which validation subset is requested. Individual probability files
    are keyed by physical image, so a smaller tuning subset and the full
    validation split share the maps they have in common.

    ``dataset`` and ``indices`` remain arguments for API compatibility with the
    first cache implementation and to make call sites explicit.
    """
    del dataset, indices
    checkpoint_sha256 = _file_sha256(checkpoint)
    manifest = {
        "cache_schema": 2,
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": checkpoint_sha256,
        "tile_size": int(tile_size),
        "tta": bool(tta),
    }
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(payload).hexdigest()[:16]
    return f"{checkpoint.stem}-{digest}", manifest


def _legacy_probability_maps(
    cache_root: Path,
    checkpoint_sha256: str,
    tile_size: int,
    tta: bool,
) -> dict[str, Path]:
    """Index maps written by the old subset-specific cache implementation."""
    found: dict[str, Path] = {}
    if not cache_root.exists():
        return found

    for directory in cache_root.iterdir():
        if not directory.is_dir():
            continue
        manifest_path = directory / "manifest.json"
        if not manifest_path.exists():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        record_keys = manifest.get("record_keys")
        if not isinstance(record_keys, list):
            continue
        if manifest.get("checkpoint_sha256") != checkpoint_sha256:
            continue
        if int(manifest.get("tile_size", -1)) != int(tile_size):
            continue
        if bool(manifest.get("tta", False)) != bool(tta):
            continue

        for position, old_key in enumerate(record_keys):
            path = directory / f"{position:05d}.npy"
            if not path.exists():
                continue
            # Old keys were ``image_id|group_key``. The group key is the
            # physical observation and therefore the reusable inference input.
            group_key = str(old_key).split("|", 1)[-1]
            found.setdefault(group_key, path)
    return found


def _link_or_copy(source: Path, destination: Path) -> None:
    """Reuse an old cached map without duplicating bytes when hard links work."""
    if destination.exists():
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def cache_probabilities(
    checkpoint: Path,
    dataset: MagfiloDataset,
    indices: list[int],
    cache_root: Path,
    device: str,
    tile_size: int,
    tta: bool,
) -> list[Path]:
    """Cache probability maps once per physical image and reuse them across subsets."""
    from filaseg.inference import InferenceConfig, predict_probability
    from filaseg.train import load_model

    namespace, manifest = probability_cache_namespace(
        checkpoint, dataset, indices, tile_size, tta
    )
    cache_dir = cache_root / namespace
    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    keys = [_record_cache_key(dataset, index) for index in indices]
    paths = [cache_dir / _record_cache_name(key) for key in keys]

    legacy = _legacy_probability_maps(
        cache_root,
        manifest["checkpoint_sha256"],
        tile_size,
        tta,
    )
    migrated = 0
    for key, path in zip(keys, paths):
        if path.exists() or key not in legacy:
            continue
        _link_or_copy(legacy[key], path)
        migrated += 1
    if migrated:
        print(f"reused {migrated} requested records from legacy probability caches")

    # Multiple annotation records can point to the same physical observation.
    # Infer one representative record per still-missing output path.
    missing_by_path: dict[Path, int] = {}
    for index, path in zip(indices, paths):
        if not path.exists():
            missing_by_path.setdefault(path, index)

    unique_requested = len(set(paths))
    cached_unique = unique_requested - len(missing_by_path)
    if not missing_by_path:
        print(
            f"reusing {unique_requested} cached physical-image probability maps "
            f"for {len(indices)} annotation records from {cache_dir}"
        )
        return paths

    print(f"probability cache: {cache_dir}")
    print(
        f"cached physical images: {cached_unique}/{unique_requested}; "
        f"new inference needed: {len(missing_by_path)}"
    )
    model, _ = load_model(checkpoint, device)
    config = InferenceConfig(tile_size=tile_size, tta=tta, device=device)
    started = time.time()
    missing = list(missing_by_path.items())
    for completed, (path, index) in enumerate(missing, start=1):
        prepared = dataset[index]
        probability = predict_probability(model, prepared.input_stack(), config)
        np.save(path, (probability * prepared.valid).astype(np.float16))
        if completed % 10 == 0 or completed == len(missing):
            print(
                f"  inference {completed}/{len(missing)}  "
                f"({time.time() - started:.0f}s)",
                flush=True,
            )
    return paths


def score(
    dataset: MagfiloDataset,
    indices: list[int],
    paths: list[Path],
    config: InstanceConfig,
) -> dict[str, float]:
    """Score one post-processing configuration over cached probability maps."""
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

    summed = {
        "one_to_many", "many_to_one", "missed", "spurious",
        "pq_tp", "pq_fp", "pq_fn",
    }
    out: dict[str, float] = {}
    for key in rows[0]:
        values = [r[key] for r in rows]
        out[key] = float(np.sum(values)) if key in summed else float(np.mean(values))
    return out


def _parameter_grid(args: argparse.Namespace) -> list[tuple[float, float, float, float]]:
    """Build the sweep and discard confidence settings that cannot filter anything."""
    grid: list[tuple[float, float, float, float]] = []
    for threshold, confidence, gap, area in itertools.product(
        args.thresholds, args.min_confidence, args.merge_gap, args.min_area_fraction
    ):
        # Every component pixel is already >= threshold. Therefore a positive
        # mean-confidence floor <= threshold cannot remove any instance and only
        # duplicates work. Zero remains the explicit "filter off" value.
        if confidence > 0 and confidence <= threshold:
            continue
        grid.append((threshold, confidence, gap, area))
    return grid


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", type=Path, dest="data_dir", default=Path("data"))
    parser.add_argument("--annotations", type=Path)
    parser.add_argument("--image-dir", type=Path, dest="image_dir")
    parser.add_argument(
        "--cache-dir", type=Path, dest="cache_dir", default=Path("data/cache")
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--prob-cache", type=Path, dest="prob_cache", default=Path("runs/prob_cache")
    )
    parser.add_argument(
        "--limit", type=int, default=60,
        help="number of records from the true grouped validation split; 0 = all",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--tile-size", type=int, default=None, dest="tile_size",
        help="inference tile; default is checkpoint config val_tile",
    )
    parser.add_argument("--tta", action="store_true")
    parser.add_argument(
        "--metric", type=str, default="matched_dice", help="metric to maximise"
    )
    parser.add_argument("--out", type=Path)

    # Focused defaults are based on the first E20 sweep. They bracket the useful
    # region while avoiding settings that were already shown to be destructive.
    parser.add_argument(
        "--thresholds", type=float, nargs="+",
        default=[0.90, 0.92, 0.93, 0.94, 0.95],
    )
    parser.add_argument(
        "--min-confidence", type=float, nargs="+", dest="min_confidence",
        default=[0.0],
        help="mean instance confidence floors; positive values <= threshold are skipped",
    )
    parser.add_argument(
        "--merge-gap", type=float, nargs="+", dest="merge_gap",
        default=[30.0, 35.0, 40.0, 45.0],
    )
    parser.add_argument(
        "--min-area-fraction", type=float, nargs="+", dest="min_area_fraction",
        default=[8e-5, 1e-4, 1.2e-4, 1.5e-4],
    )
    args = parser.parse_args()

    if args.data_dir and not args.image_dir:
        args.image_dir = discover(args.data_dir).train_dir
    annotations = resolve_annotations(args.annotations, args.image_dir, args.data_dir)
    checkpoint_blob = _checkpoint_blob(args.checkpoint)
    checkpoint_config = checkpoint_blob.get("config", {}) or {}
    if args.tile_size is None:
        args.tile_size = int(checkpoint_config.get("val_tile", 512))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset, indices, split_info = validation_dataset(
            annotations,
            args.image_dir,
            args.cache_dir,
            checkpoint_config,
            args.limit,
        )

    print(
        "validation split from checkpoint: "
        f"seed={split_info['seed']}  val_fraction={split_info['val_fraction']:.3f}"
    )
    print(
        f"full held-out: {split_info['full_validation_records']} annotation records / "
        f"{split_info['full_validation_groups']} physical images"
    )
    print(
        f"tuning on: {split_info['selected_records']} annotation records / "
        f"{split_info['selected_groups']} physical images"
    )
    print(f"inference: tile={args.tile_size}  tta={args.tta}  device={args.device}\n")

    paths = cache_probabilities(
        args.checkpoint,
        dataset,
        indices,
        args.prob_cache,
        args.device,
        args.tile_size,
        args.tta,
    )

    grid = _parameter_grid(args)
    if not grid:
        raise SystemExit("parameter grid is empty after removing redundant settings")
    print(f"\nsweeping {len(grid)} configurations over the cached maps")
    print(
        f"{'thr':>7}{'conf':>7}{'gap':>7}{'minarea':>11}"
        f"{'matchedD':>10}{'fgDice':>8}{'PQ':>7}{'RQ':>7}"
        f"{'inst':>7}{'spur':>7}{'1->m':>6}{'miss':>6}"
    )

    results: list[dict] = []
    for threshold, confidence, gap, area in grid:
        config = InstanceConfig(
            threshold=threshold,
            min_confidence=confidence,
            merge_gap=gap,
            min_area_fraction=area,
        )
        summary = score(dataset, indices, paths, config)
        results.append(
            {
                "threshold": threshold,
                "min_confidence": confidence,
                "merge_gap": gap,
                "min_area_fraction": area,
                **summary,
            }
        )
        print(
            f"{threshold:>7.3f}{confidence:>7.3f}{gap:>7.1f}{area:>11.2e}"
            f"{summary['matched_dice']:>10.4f}{summary['foreground_dice']:>8.4f}"
            f"{summary['pq']:>7.4f}{summary['rq']:>7.4f}"
            f"{summary['n_instances']:>7.1f}{summary['spurious']:>7.0f}"
            f"{summary['one_to_many']:>6.0f}{summary['missed']:>6.0f}",
            flush=True,
        )

    best = max(results, key=lambda row: row.get(args.metric, 0.0))
    print("\n" + "=" * 74)
    print(f"BEST by {args.metric}")
    print("=" * 74)
    parameter_keys = {
        "threshold": ".6f",
        "min_confidence": ".6f",
        "merge_gap": ".3f",
        "min_area_fraction": ".8g",
    }
    for key in (
        "threshold", "min_confidence", "merge_gap", "min_area_fraction",
        "matched_dice", "matched_dice_over_truth", "matched_dice_over_pred",
        "mean_paired_dice", "foreground_dice", "pq", "rq", "n_instances",
        "spurious", "one_to_many", "many_to_one", "missed",
    ):
        if key not in best:
            continue
        if key in parameter_keys:
            value = format(best[key], parameter_keys[key])
        else:
            value = f"{best[key]:.4f}"
        print(f"  {key:24s} {value}")

    print("\nUse it with:")
    print(
        f"  python scripts/predict.py --images data/test \\\n"
        f"      --checkpoint {args.checkpoint} --out submission.csv \\\n"
        f"      --threshold {best['threshold']} "
        f"--min-confidence {best['min_confidence']} \\\n"
        f"      --merge-gap {best['merge_gap']} "
        f"--min-area-fraction {best['min_area_fraction']} \\\n"
        f"      --tile-size {args.tile_size} --no-tta"
    )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "checkpoint": str(args.checkpoint),
            "tile_size": args.tile_size,
            "tta": args.tta,
            "split": split_info,
            "best": best,
            "all": results,
        }
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
