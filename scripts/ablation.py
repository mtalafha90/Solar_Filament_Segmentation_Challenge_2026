#!/usr/bin/env python3
"""Train several model variants and compare them, to show what each piece buys.

Each variant differs from the full model in exactly one respect, so the
difference in score is attributable.  Run it on the real dataset to justify the
design choices, or on synthetic data for a quick smoke test::

    python scripts/ablation.py --annotations data/magfilo/annotations.json \
        --image-dir data/magfilo/images --epochs 40
"""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from pathlib import Path

import _bootstrap  # noqa: F401

from filaseg.losses import LossWeights
from filaseg.models.filanet import FilaNetConfig
from filaseg.train import TrainConfig, train

VARIANTS: dict[str, dict] = {
    "full": {},
    "no_cl_dice": {"loss": {"cl_dice": 0.0}},
    "no_edge_attention": {"model": {"edge_attention": False}},
    "no_aux_heads": {
        "model": {"aux_heads": False},
        "loss": {"spine": 0.0, "boundary": 0.0},
    },
    "no_deep_supervision": {
        "model": {"deep_supervision": False},
        "loss": {"deep": 0.0},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True, dest="image_dir")
    parser.add_argument("--cache-dir", type=Path, dest="cache_dir")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/ablation"),
                        dest="output_dir")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--patch-size", type=int, default=256, dest="patch_size")
    parser.add_argument("--batch-size", type=int, default=8, dest="batch_size")
    parser.add_argument("--samples-per-epoch", type=int, default=1000,
                        dest="samples_per_epoch")
    parser.add_argument("--learning-rate", type=float, default=6e-4, dest="learning_rate")
    parser.add_argument("--base-width", type=int, default=32, dest="base_width")
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS))
    args = parser.parse_args()

    results: dict[str, dict] = {}
    for name in args.variants:
        if name not in VARIANTS:
            raise SystemExit(f"unknown variant {name!r}; choose from {list(VARIANTS)}")
        overrides = VARIANTS[name]
        model = FilaNetConfig(base_width=args.base_width, depth=args.depth,
                              **overrides.get("model", {}))
        loss = LossWeights(**overrides.get("loss", {}))

        config = TrainConfig(
            annotations=str(args.annotations),
            image_dir=str(args.image_dir),
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            output_dir=str(args.output_dir / name),
            epochs=args.epochs,
            patch_size=args.patch_size,
            batch_size=args.batch_size,
            samples_per_epoch=args.samples_per_epoch,
            learning_rate=args.learning_rate,
            device=args.device,
            seed=args.seed,
            model=model,
            loss=loss,
        )
        print(f"\n{'=' * 60}\nvariant: {name}\n{'=' * 60}", flush=True)
        results[name] = train(config)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "ablation.json").open("w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"\n{'=' * 74}")
    print(f"{'variant':<22}{'IoU':>10}{'Dice':>10}{'clDice':>10}{'MSIoU':>10}{'thr':>7}")
    print("=" * 74)
    baseline = results.get("full", {}).get("best_iou")
    for name, summary in results.items():
        delta = ""
        if baseline is not None and name != "full":
            delta = f"  ({summary['best_iou'] - baseline:+.4f})"
        print(f"{name:<22}{summary['best_iou']:>10.4f}{summary['best_dice']:>10.4f}"
              f"{summary['best_cl_dice']:>10.4f}{summary['best_msiou']:>10.4f}"
              f"{summary['best_threshold']:>7.2f}{delta}")


if __name__ == "__main__":
    main()
