"""Training loop for FilaNet."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data.dataset import FilamentPatchDataset, MagfiloDataset
from .inference import InferenceConfig, predict_probability
from .losses import FilamentLoss, LossWeights
from .metrics import (
    aggregate,
    cl_dice,
    fragmentation,
    instance_masks_from_labels,
    multiscale_iou,
    panoptic_quality,
    pixel_scores,
)
from .models.filanet import FilaNet, FilaNetConfig, build_model
from .postprocess.instances import InstanceConfig, extract_instances


@dataclass
class TrainConfig:
    """Everything that controls a training run."""

    annotations: str = ""
    image_dir: str = ""
    cache_dir: str | None = None
    output_dir: str = "runs/filanet"

    val_fraction: float = 0.15
    seed: int = 0

    patch_size: int = 256
    batch_size: int = 8
    samples_per_epoch: int = 2000
    positive_fraction: float = 0.7
    num_workers: int = 0

    epochs: int = 60
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 2
    grad_clip: float = 1.0
    amp: bool = True

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model: FilaNetConfig = field(default_factory=FilaNetConfig)
    loss: LossWeights = field(default_factory=LossWeights)
    pos_weight: float = 4.0

    val_every: int = 1
    val_tile: int = 512
    thresholds: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)
    """Thresholds tried on the validation set; the best one is stored with the model."""
    selection_metric: str = "pq"
    """Metric used to pick the threshold and the best checkpoint.

    Defaults to Panoptic Quality, because that is what the challenge ranks on
    and it is not interchangeable with pixel overlap: a model that splits every
    filament in two can hold its Dice score while halving its PQ. Set to "dice"
    for the other primary criterion, or "iou" for plain pixel overlap, which is
    the cheapest to compute.
    """
    instance_config: InstanceConfig = field(default_factory=InstanceConfig)
    """Post-processing used when validating, so validation matches submission."""
    instance_thresholds: int = 3
    """How many thresholds to evaluate the costly instance metrics at."""


def split_ids(
    n_images: int, val_fraction: float, seed: int
) -> tuple[list[int], list[int]]:
    """Split indices into training and validation sets, deterministically."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_images)
    n_val = max(1, int(round(n_images * val_fraction))) if n_images > 1 else 0
    return sorted(order[n_val:].tolist()), sorted(order[:n_val].tolist())


def build_scheduler(
    optimiser: torch.optim.Optimizer, config: TrainConfig, steps_per_epoch: int
):
    """Linear warmup into a cosine decay.

    Warmup matters here because the loss mixes terms with very different
    curvature; stepping straight to the full learning rate tends to collapse the
    prediction to all-background in the first few hundred steps.
    """
    warmup_steps = max(1, config.warmup_epochs * steps_per_epoch)
    total_steps = max(warmup_steps + 1, config.epochs * steps_per_epoch)

    def factor(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimiser, factor)


@torch.no_grad()
def validate(
    model: FilaNet,
    dataset: MagfiloDataset,
    indices: list[int],
    thresholds: tuple[float, ...],
    device: str,
    tile_size: int = 512,
    tta: bool = False,
    selection_metric: str = "pq",
    instance_config: InstanceConfig | None = None,
    instance_thresholds: int = 3,
) -> dict[str, float]:
    """Score the model on whole validation frames, at several thresholds.

    Validation runs on full disks, with the same post-processing a submission
    would use, because that is what the challenge measures. Patch-level numbers
    flatter a model that relies on every crop being centred on a filament, and
    pixel-level numbers say nothing about whether filaments came out as single
    objects -- which is half of what Panoptic Quality is measuring.

    Args:
        model: The network to score.
        dataset: Validation observations.
        indices: Which observations to use.
        thresholds: Probability thresholds to try.
        device: Torch device.
        tile_size: Tile size for whole-disk inference.
        tta: Apply test-time augmentation.
        selection_metric: Which metric picks the best threshold.
        instance_config: Post-processing settings.
        instance_thresholds: How many of the best thresholds by pixel overlap to
            evaluate the expensive instance metrics at.

    Returns:
        A flat dictionary of per-threshold and best-threshold scores.
    """
    model.eval()
    config = InferenceConfig(tile_size=tile_size, tta=tta, device=device)
    instance_config = instance_config or InstanceConfig()

    per_threshold: dict[float, list[dict[str, float]]] = {t: [] for t in thresholds}

    # Instance extraction is far more expensive than thresholding -- it
    # skeletonises every component to decide what to merge -- so running it at
    # every threshold on full-resolution frames would dominate the epoch. Pixel
    # metrics are computed everywhere, and the instance metrics only at the few
    # thresholds that look best by pixel overlap. The optimum of PQ is never far
    # from the optimum of Dice, so this costs nothing in practice.
    cheap_key = "dice" if selection_metric != "iou" else "iou"
    probabilities: list[tuple[np.ndarray, object]] = []

    for index in indices:
        prepared = dataset[index]
        probability = predict_probability(model, prepared.input_stack(), config)
        probability = probability * prepared.valid
        probabilities.append((probability, prepared))

        for threshold in thresholds:
            predicted = probability >= threshold
            scores = pixel_scores(predicted, prepared.mask, prepared.valid).as_dict()
            scores["cl_dice"] = cl_dice(predicted, prepared.mask)
            scores["msiou"] = float(multiscale_iou(predicted, prepared.mask))
            per_threshold[threshold].append(scores)

    ranked = sorted(
        thresholds,
        key=lambda t: aggregate(per_threshold[t]).get(cheap_key, 0.0),
        reverse=True,
    )
    candidates = ranked[: max(1, instance_thresholds)]

    # Attach instance metrics to the matching per-image records.
    for position, (probability, prepared) in enumerate(probabilities):
        truth_instances = instance_masks_from_labels(prepared.instances)
        for threshold in candidates:
            labels = extract_instances(
                probability,
                prepared.valid,
                replace(instance_config, threshold=threshold),
            )
            predicted_instances = instance_masks_from_labels(labels)
            record = per_threshold[threshold][position]
            record.update(
                panoptic_quality(predicted_instances, truth_instances).as_dict()
            )
            record.update(fragmentation(predicted_instances, truth_instances).as_dict())

    key = {"pq": "pq", "dice": "dice", "iou": "iou"}.get(selection_metric, "pq")
    summary: dict[str, float] = {}
    best_threshold, best_value = candidates[0], -1.0
    reported = ("iou", "dice", "cl_dice", "msiou", "precision", "recall", "pq", "sq", "rq")
    for threshold, records in per_threshold.items():
        merged = aggregate(records)
        # Only thresholds carrying instance metrics can win on PQ.
        if key == "pq" and threshold not in candidates:
            for name in reported:
                summary[f"{name}@{threshold:.2f}"] = merged.get(name, 0.0)
            continue
        if merged.get(key, 0.0) > best_value:
            best_value = merged[key]
            best_threshold = threshold
        for name in reported:
            summary[f"{name}@{threshold:.2f}"] = merged.get(name, 0.0)

    summary["best_threshold"] = float(best_threshold)
    best = aggregate(per_threshold[best_threshold])
    for name in (*reported, "one_to_many", "many_to_one", "missed", "spurious"):
        summary[f"best_{name}"] = best.get(name, 0.0)
    summary["best_value"] = float(best_value)
    # Kept for backwards compatibility with earlier checkpoints and logs.
    summary["best_iou"] = best.get("iou", 0.0)
    return summary


def train(config: TrainConfig) -> dict[str, float]:
    """Train FilaNet and write checkpoints, logs and a config record.

    Args:
        config: The run configuration.

    Returns:
        The best validation summary seen during the run.
    """
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    source = MagfiloDataset(config.annotations, config.image_dir, config.cache_dir)
    if len(source) == 0:
        raise ValueError("the dataset is empty; check --annotations and --image-dir")
    train_indices, val_indices = split_ids(len(source), config.val_fraction, config.seed)

    train_source = MagfiloDataset(
        config.annotations,
        config.image_dir,
        config.cache_dir,
        image_ids=[source.records[i].image_id for i in train_indices],
    )
    patches = FilamentPatchDataset(
        train_source,
        patch_size=config.patch_size,
        samples_per_epoch=config.samples_per_epoch,
        positive_fraction=config.positive_fraction,
        augment=True,
        seed=config.seed,
    )
    loader = DataLoader(
        patches,
        batch_size=config.batch_size,
        shuffle=False,  # the dataset already samples at random
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        drop_last=True,
    )

    device = torch.device(config.device)
    model = build_model(config.model).to(device)
    criterion = FilamentLoss(config.loss, pos_weight=config.pos_weight).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = build_scheduler(optimiser, config, max(1, len(loader)))
    use_amp = config.amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    with (output_dir / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(_config_to_dict(config), handle, indent=2)

    history: list[dict] = []
    best: dict[str, float] = {"best_value": -1.0}

    for epoch in range(config.epochs):
        model.train()
        patches.set_epoch(epoch)
        started = time.time()
        running: dict[str, float] = {}
        n_batches = 0

        for batch in loader:
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            optimiser.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(batch["input"])
                loss, components = criterion(outputs, batch)

            scaler.scale(loss).backward()
            if config.grad_clip > 0:
                scaler.unscale_(optimiser)
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimiser)
            scaler.update()
            scheduler.step()

            for key, value in components.items():
                running[key] = running.get(key, 0.0) + value
            n_batches += 1

        record = {
            "epoch": epoch,
            "seconds": round(time.time() - started, 1),
            "lr": scheduler.get_last_lr()[0],
            **{f"train_{k}": v / max(1, n_batches) for k, v in running.items()},
        }

        if val_indices and (epoch + 1) % config.val_every == 0:
            val_source = MagfiloDataset(
                config.annotations,
                config.image_dir,
                config.cache_dir,
                image_ids=[source.records[i].image_id for i in val_indices],
            )
            summary = validate(
                model,
                val_source,
                list(range(len(val_source))),
                config.thresholds,
                config.device,
                config.val_tile,
                selection_metric=config.selection_metric,
                instance_config=config.instance_config,
                instance_thresholds=config.instance_thresholds,
            )
            record.update({f"val_{k}": v for k, v in summary.items()})

            if summary["best_value"] > best.get("best_value", -1.0):
                best = summary
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": _config_to_dict(config),
                        "epoch": epoch,
                        "val": summary,
                        "threshold": summary["best_threshold"],
                    },
                    output_dir / "best.pt",
                )

        history.append(record)
        with (output_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(history, handle, indent=2)

        message = (
            f"epoch {epoch + 1}/{config.epochs}  "
            f"loss {record.get('train_total', float('nan')):.4f}  "
            f"({record['seconds']}s)"
        )
        if "val_best_pq" in record:
            message += (
                f"  val PQ {record['val_best_pq']:.4f}"
                f"  Dice {record.get('val_best_dice', 0):.4f}"
                f"  IoU {record.get('val_best_iou', 0):.4f}"
                f"  clDice {record.get('val_best_cl_dice', 0):.4f}"
                f"  @thr {record['val_best_threshold']:.2f}"
            )
        print(message, flush=True)

    torch.save(
        {
            "model": model.state_dict(),
            "config": _config_to_dict(config),
            "epoch": config.epochs - 1,
            "val": best,
            "threshold": best.get("best_threshold", 0.5),
        },
        output_dir / "last.pt",
    )
    return best


def _config_to_dict(config: TrainConfig) -> dict:
    """Convert a config to plain JSON-serialisable types."""
    raw = asdict(config)
    return json.loads(json.dumps(raw, default=str))


def load_model(checkpoint_path: str | Path, device: str = "cpu") -> tuple[FilaNet, float]:
    """Load a trained model and the validation-calibrated threshold that goes with it."""
    blob = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = blob.get("config", {}).get("model", {})
    model = build_model(model_config if isinstance(model_config, dict) else None)
    model.load_state_dict(blob["model"])
    model.to(device).eval()
    return model, float(blob.get("threshold", 0.5))
