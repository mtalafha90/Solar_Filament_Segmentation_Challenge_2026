# Corrected post-processing tuning protocol

**Date:** 2026-08-21 (Asia/Dubai)

This note records the corrections applied after the first CPU E20 post-processing sweep and the first corrected true-validation tuning result.

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

## Corrected CPU E20 60-record sweep

Command:

```bash
/usr/bin/time -v python scripts/tune_postprocess.py \
    --data-dir data \
    --checkpoint runs/cpu_filanet_20epoch/best.pt \
    --limit 60 \
    --out runs/cpu_filanet_20epoch/postprocess_trueval_60.json
```

Setup actually used:

- grouped split: seed 0, validation fraction 0.15;
- full validation: 168 annotation records / 106 physical images;
- selected tuning subset: 60 annotation records / 57 physical images;
- inference tile: 256;
- TTA: off;
- device: CPU;
- probability cache: `runs/prob_cache/best-fc663d58e2ec7932`;
- configurations: 80;
- inference time for the 60 maps: 3325 s;
- total wall time: 2:22:17;
- maximum RSS: 1,341,188 kB;
- swaps: 0;
- exit status: 0.

### Best configuration

| parameter / metric | value |
|---|---:|
| threshold | **0.94** |
| min confidence | **0.00** |
| merge gap | **40.0** |
| min-area fraction | **1.5e-4** |
| matched Dice | **0.3128** |
| matched Dice over truth | 0.5174 |
| matched Dice over prediction | 0.3887 |
| mean paired Dice | 0.6426 |
| foreground Dice | 0.5396 |
| PQ | 0.2292 |
| RQ | 0.3598 |
| mean instances / observation | 8.9833 |
| spurious | 185 |
| one-to-many | 40 |
| many-to-one | 15 |
| missed | 112 |

The command emitted by the tuner for this setting is:

```bash
python scripts/predict.py --images data/test \
    --checkpoint runs/cpu_filanet_20epoch/best.pt --out submission.csv \
    --threshold 0.94 --min-confidence 0.0 \
    --merge-gap 40.0 --min-area-fraction 0.00015 \
    --tile-size 256 --no-tta
```

This is **not yet Submission 2**. It is the current best post-processing setting on a deterministic 60-record subset of the true grouped validation set.

## Interpretation

1. The corrected optimum is a broad plateau rather than an isolated point. At `min_area_fraction=1.5e-4`, matched Dice remains near 0.31 across thresholds 0.93--0.95 and merge gaps around 30--45.
2. The largest tested area floor, `1.5e-4`, is the best area setting throughout the high-threshold region. Because it is the upper edge of the grid, the optimum is not yet bracketed; the next sweep should extend the area floor upward before committing to a full 168-record validation.
3. `merge_gap=40` remains a good central choice. Changes of +/-5 pixels have modest effects compared with the effect of the area floor.
4. The best foreground Dice (0.5396) is almost unchanged from the earlier semantic baseline, while matched Dice rises only to 0.3128. This confirms that instance construction/counting remains the main loss mechanism.
5. Mean paired Dice is 0.6426, much higher than matched Dice 0.3128. When a predicted filament is successfully matched, its shape overlap is substantially better than the final instance-level score; unmatched/spurious/missed structures still dominate the penalty.
6. Relative to the earlier provisional 0.3005 result, the corrected best is numerically higher by about 4.1%, but the two numbers are **not a controlled comparison** because both the validation subset and tile size changed.

## Next refinement on the same cached 60 maps

Before spending CPU on all 168 held-out annotation records, extend only the parameter dimension that is still hitting a grid boundary. Recommended focused refinement:

```bash
python scripts/tune_postprocess.py --data-dir data \
    --checkpoint runs/cpu_filanet_20epoch/best.pt \
    --limit 60 \
    --thresholds 0.93 0.94 0.95 \
    --merge-gap 35 40 45 \
    --min-area-fraction 0.00015 0.00017 0.00019 0.00021 0.00023 \
    --min-confidence 0 \
    --out runs/cpu_filanet_20epoch/postprocess_trueval_60_refined.json
```

This is 45 post-processing configurations and should reuse the existing `best-fc663d58e2ec7932` probability cache, so no neural inference should be repeated.

After this refined sweep brackets the min-area optimum, run only the best small set of candidate configurations on all 168 held-out annotation records. The resulting full-split winner should determine Submission 2.
