# Corrected post-processing tuning protocol

**Date:** 2026-08-21 (Asia/Dubai)

This note records the corrections applied after the first CPU E20 post-processing sweep and the subsequent true-validation tuning experiments.

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

### First corrected best configuration

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

The best area value was the upper edge of the first corrected grid, so the area floor was extended before full-split validation.

## Refined 60-record sweep

Command:

```bash
/usr/bin/time -v python scripts/tune_postprocess.py \
    --data-dir data \
    --checkpoint runs/cpu_filanet_20epoch/best.pt \
    --limit 60 \
    --thresholds 0.93 0.94 0.95 \
    --merge-gap 35 40 45 \
    --min-area-fraction 0.00015 0.00017 0.00019 0.00021 0.00023 \
    --min-confidence 0 \
    --out runs/cpu_filanet_20epoch/postprocess_trueval_60_refined.json
```

The run correctly reused all 60 cached probability maps from `runs/prob_cache/best-fc663d58e2ec7932`, so no neural inference was repeated. It swept 45 post-processing configurations in 47:46.65 wall time, with maximum RSS 1,023,800 kB, zero swaps, and exit status 0.

### Refined best configuration

| parameter / metric | value |
|---|---:|
| threshold | **0.93** |
| min confidence | **0.00** |
| merge gap | **40.0** |
| min-area fraction | **2.3e-4** |
| matched Dice | **0.3398** |
| matched Dice over truth | 0.4756 |
| matched Dice over prediction | 0.4551 |
| mean paired Dice | 0.6421 |
| foreground Dice | 0.5265 |
| PQ | 0.2471 |
| RQ | 0.3804 |
| mean instances / observation | 6.7333 |
| spurious | 99 |
| one-to-many | 26 |
| many-to-one | 11 |
| missed | 151 |

Relative to the first corrected best (`matched_dice=0.3128`), the refined best rises to 0.3398, an 8.6% relative increase. Spurious detections fall from 185 to 99 and mean predicted instances fall from 8.98 to 6.73 per observation. The trade-off is increased misses (112 to 151) and a modest fall in foreground Dice (0.5396 to 0.5265).

### Parameter-response evidence

At threshold 0.93 and merge gap 40, the area sweep is still monotonic over the tested range:

| min-area fraction | matched Dice | foreground Dice | mean instances | spurious | missed |
|---:|---:|---:|---:|---:|---:|
| 1.5e-4 | 0.3115 | 0.5402 | 9.4 | 208 | 106 |
| 1.7e-4 | 0.3230 | 0.5413 | 8.6 | 173 | 114 |
| 1.9e-4 | 0.3269 | 0.5375 | 8.1 | 156 | 131 |
| 2.1e-4 | 0.3351 | 0.5297 | 7.2 | 117 | 142 |
| **2.3e-4** | **0.3398** | 0.5265 | **6.7** | **99** | 151 |

At threshold 0.94 the curve is already bracketed around roughly 1.7e-4--2.1e-4, and at threshold 0.95 it peaks around roughly 2.1e-4 before dropping sharply at 2.3e-4. The only remaining grid-boundary problem is therefore the threshold-0.93 branch.

`merge_gap=40` remains the strongest central value throughout the refined grid. Values 35 and 45 are close but consistently slightly lower near the best region, so further refinement should hold the merge gap at 40 and spend the remaining search budget on threshold/area.

## Interpretation

1. Instance filtering is materially improving the competition-aligned proxy without changing the neural network.
2. The main gain comes from suppressing small false components: matched Dice improves while foreground Dice falls slightly, demonstrating that union-mask overlap and per-filament scoring prefer different operating points.
3. The refined mean prediction count of 6.73 instances per observation is much closer to the MAGFiLO dataset-wide annotation density (~7.1 filaments per annotation record) than the original test submission (~11.9 predicted instances per image). This is supportive evidence, not a scoring target by itself.
4. Mean paired Dice remains almost unchanged (~0.642), while matched Dice improves from 0.313 to 0.340. Thus the improvement is almost entirely from better instance selection/counting rather than boundary quality.
5. The current best is still a tuning-subset result, not a final held-out estimate and not a Kaggle score.

## One final 60-record bracket before full validation

Because threshold 0.93 still improves monotonically at the largest tested area floor, do one last narrow sweep using the same cached probability maps. Hold merge gap at 40 and resolve the threshold/area interaction around the boundary. Recommended command:

```bash
python scripts/tune_postprocess.py --data-dir data \
    --checkpoint runs/cpu_filanet_20epoch/best.pt \
    --limit 60 \
    --thresholds 0.91 0.92 0.925 0.93 0.935 0.94 \
    --merge-gap 40 \
    --min-area-fraction 0.00021 0.00023 0.00025 0.00027 0.00029 0.00031 \
    --min-confidence 0 \
    --out runs/cpu_filanet_20epoch/postprocess_trueval_60_finalbracket.json
```

This is 36 post-processing configurations and should reuse the existing 60 probability maps with no neural inference. Once this sweep places the optimum inside the tested area range, carry only the best few configurations to the full 168-record grouped validation split. The full-split winner will determine Submission 2.
