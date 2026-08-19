"""Training loop for FilaNet."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data.dataset import FilamentPatchDataset, MagfiloDataset
from .inference import InferenceConfig, predict_probability
from .losses import FilamentLoss, LossWeights
from .metrics import aggregate, cl_dice, multiscale_iou, pixel_scores
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
) -> dict[str, float]:
    """Score the model on whole validation frames, at several thresholds.

    Validation runs on full disks rather than on patches because that is how the
    model is scored in the end: patch-level numbers flatter a model that relies
    on every crop being centred on a filament.
    """
    model.eval()
    config = InferenceConfig(tile_size=tile_size, tta=tta, device=device)

    per_threshold: dict[float, list[dict[str, float]]] = {t: [] for t in thresholds}
    for index in indices:
        prepared = dataset[index]
        probability = predict_probability(model, prepared.input_stack(), config)
        probability = probability * prepared.valid
        truth = prepared.mask
        for threshold in thresholds:
            predicted = probability >= threshold
            scores = pixel_scores(predicted, truth, prepared.valid).as_dict()
            scores["cl_dice"] = cl_dice(predicted, truth)
            scores["msiou"] = float(multiscale_iou(predicted, truth))
            per_threshold[threshold].append(scores)

    summary: dict[str, float] = {}
    best_threshold, best_iou = thresholds[0], -1.0
    for threshold, records in per_threshold.items():
        merged = aggregate(records)
        if merged.get("iou", 0.0) > best_iou:
            best_iou = merged["iou"]
            best_threshold = threshold
        for key in ("iou", "dice", "cl_dice", "msiou", "precision", "recall"):
            summary[f"{key}@{threshold:.2f}"] = merged.get(key, 0.0)

    summary["best_threshold"] = float(best_threshold)
    summary["best_iou"] = float(best_iou)
    for key in ("dice", "cl_dice", "msiou", "precision", "recall"):
        summary[f"best_{key}"] = summary.get(f"{key}@{best_threshold:.2f}", 0.0)
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
    best: dict[str, float] = {"best_iou": -1.0}

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
            )
            record.update({f"val_{k}": v for k, v in summary.items()})

            if summary["best_iou"] > best.get("best_iou", -1.0):
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
        if "val_best_iou" in record:
            message += (
                f"  val IoU {record['val_best_iou']:.4f}"
                f"  clDice {record.get('val_best_cl_dice', 0):.4f}"
                f"  MSIoU {record.get('val_best_msiou', 0):.4f}"
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
