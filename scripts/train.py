#!/usr/bin/env python3
"""Train FilaNet on a MAGFiLO-style dataset.

Example::

    python scripts/train.py \
        --annotations data/magfilo/annotations.json \
        --image-dir data/magfilo/images \
        --cache-dir data/cache \
        --output-dir runs/filanet \
        --epochs 60 --patch-size 256 --batch-size 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401

import yaml

from filaseg.data.layout import discover, resolve_annotations
from filaseg.losses import LossWeights
from filaseg.models.filanet import FilaNetConfig
from filaseg.train import TrainConfig, train


def build_config(args: argparse.Namespace) -> TrainConfig:
    settings: dict = {}
    if args.config:
        with Path(args.config).open("r", encoding="utf-8") as handle:
            settings = yaml.safe_load(handle) or {}

    model_settings = settings.pop("model", {}) or {}
    loss_settings = settings.pop("loss", {}) or {}

    # Explicit command-line arguments override the config file.
    for name, value in vars(args).items():
        if name in ("config",) or value is None:
            continue
        if name == "data_dir":
            continue
        if name in TrainConfig.__dataclass_fields__:
            settings[name] = value

    settings["model"] = FilaNetConfig(**model_settings)
    settings["loss"] = LossWeights(**loss_settings)
    known = set(TrainConfig.__dataclass_fields__)
    unknown = set(settings) - known
    if unknown:
        raise SystemExit(f"unknown settings in config: {sorted(unknown)}")
    return TrainConfig(**settings)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=str, help="YAML file of defaults")
    parser.add_argument("--data-dir", type=str, dest="data_dir",
                        help="dataset root holding train/ and test/; the "
                             "annotations and image directory are found inside it")
    parser.add_argument("--annotations", type=str,
                        help="annotation JSON; discovered automatically if omitted")
    parser.add_argument("--image-dir", type=str, dest="image_dir")
    parser.add_argument("--cache-dir", type=str, dest="cache_dir")
    parser.add_argument("--output-dir", type=str, dest="output_dir")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int, dest="batch_size")
    parser.add_argument("--patch-size", type=int, dest="patch_size")
    parser.add_argument("--samples-per-epoch", type=int, dest="samples_per_epoch")
    parser.add_argument("--learning-rate", type=float, dest="learning_rate")
    parser.add_argument("--val-fraction", type=float, dest="val_fraction")
    parser.add_argument("--num-workers", type=int, dest="num_workers")
    parser.add_argument("--device", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--no-amp", dest="amp", action="store_false", default=None)
    args = parser.parse_args()

    # Resolve the dataset before building the config, so a mistyped or omitted
    # annotation filename is corrected rather than reported as missing.
    if args.data_dir and not args.image_dir:
        layout = discover(args.data_dir)
        args.image_dir = str(layout.train_dir) if layout.train_dir else None
    try:
        args.annotations = str(
            resolve_annotations(args.annotations, args.image_dir, args.data_dir)
        )
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error

    config = build_config(args)
    if not config.image_dir:
        raise SystemExit("--image-dir is required (or pass --data-dir)")

    print(f"training on {config.device}, writing to {config.output_dir}")
    best = train(config)
    print("\nbest validation scores:")
    print(json.dumps({k: round(v, 4) for k, v in best.items()}, indent=2))


if __name__ == "__main__":
    main()
