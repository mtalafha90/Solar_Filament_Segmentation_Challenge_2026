# Corrected post-processing tuning protocol

**Date:** 2026-08-21 (Asia/Dubai)

This note records the corrections applied after the first CPU E20 post-processing sweep.

## Why the first sweep is provisional

The first sweep used:

```bash
python scripts/tune_postprocess.py --data-dir data \
    --checkpoint runs/cpu_filanet_20epoch/best.pt --limit 60
```

It reported a best provisional matched Dice of 0.3005 at threshold 0.93, merge gap 40, and min-area fraction 1.2e-4. That result remains useful for parameter-response trends, but it is **not** a valid held-out estimate for two reasons:

1. the old tuner selected the final 60 annotation records instead of reconstructing the grouped seed-0 validation split used during training;
2. the old tuner defaulted to `tile_size=512`, while the CPU E20 checkpoint was validated and submitted with `val_tile=256`.

Therefore the old cached probability maps must not be reused for the corrected experiment.

## Corrections now on `main`

- `tune_postprocess.py` reconstructs the exact grouped validation split from the checkpoint's stored `val_fraction` and `seed` using the same `split_ids(..., groups=source.group_keys)` path as training.
- `--limit` now selects a deterministic evenly spaced subset from the true validation set; `--limit 0` uses the full split.
- When `--tile-size` is omitted, the tuner inherits the checkpoint's stored `val_tile` value.
- Probability caches are namespaced by checkpoint SHA-256, selected validation records, tile size and TTA setting, with a manifest stored beside the arrays.
- Redundant positive `min_confidence <= threshold` combinations are skipped.
- The focused default grid is threshold 0.90--0.95, merge gap 30--45, and min-area fraction 8e-5--1.5e-4, with confidence filtering off for this first corrected pass.
- Training/YAML defaults now select checkpoints by `matched_dice` rather than foreground Dice.
- When `selection_metric` is `matched_dice` or `pq`, validation evaluates instance metrics at every configured threshold instead of shortlisting thresholds by foreground Dice.
- Regression tests cover grouped splitting, matched-Dice defaults, deterministic validation subsampling, redundant-confidence removal, and probability-cache identity.

## Corrected CPU E20 sweep

After updating local `main`, run:

```bash
python scripts/tune_postprocess.py --data-dir data \
    --checkpoint runs/cpu_filanet_20epoch/best.pt \
    --limit 60 \
    --out runs/cpu_filanet_20epoch/postprocess_trueval_60.json
```

Expected setup for the existing CPU E20 checkpoint:

- grouped split: seed 0, validation fraction 0.15;
- full validation: 168 annotation records / 106 physical images;
- selected tuning subset: 60 evenly spaced records from that true validation set;
- inference tile: 256;
- TTA: off;
- focused grid: 80 configurations.

The corrected result should be used to choose the next post-processing configuration. Only after that should the best narrow region be confirmed on all 168 held-out annotation records before producing another Kaggle submission.
