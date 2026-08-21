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

Therefore the old result is retained only as historical parameter-response evidence.

## Corrections on `main`

- `tune_postprocess.py` reconstructs the exact grouped validation split from the checkpoint's stored `val_fraction` and `seed` using the same `split_ids(..., groups=source.group_keys)` path as training.
- `--limit` selects a deterministic evenly spaced subset from the true validation set; `--limit 0` uses the full split.
- When `--tile-size` is omitted, the tuner inherits the checkpoint's stored `val_tile` value.
- Training/YAML defaults select checkpoints by `matched_dice` rather than foreground Dice.
- When `selection_metric` is `matched_dice` or `pq`, validation evaluates instance metrics at every configured threshold instead of shortlisting thresholds by foreground Dice.
- Redundant positive `min_confidence <= threshold` combinations are skipped.
- Probability maps are now cached per physical image under a namespace determined only by checkpoint content, tile size, and TTA. This allows a tuning subset and the full validation split to share maps, and duplicate annotation records for the same physical observation share one neural inference result.
- The cache migration path recognizes the earlier subset-specific cache manifests and hard-links/copies matching maps into the new shared cache, so the 57 physical images already inferred in the 60-record experiment can be reused when expanding to all 106 held-out physical images.
- Tuning-table threshold output now prints three decimals, avoiding ambiguity between values such as 0.925 and 0.935; best-parameter output also preserves the actual min-area value instead of rounding 2.3e-4 to 0.0002.
- Regression tests cover grouped splitting, matched-Dice defaults, deterministic validation subsampling, redundant-confidence removal, cross-subset cache reuse, duplicate physical-image cache sharing, and legacy-cache migration.

## Corrected CPU E20 60-record sweep

Setup:

- grouped split: seed 0, validation fraction 0.15;
- full validation: 168 annotation records / 106 physical images;
- selected tuning subset: 60 annotation records / 57 physical images;
- inference tile: 256;
- TTA: off;
- device: CPU;
- original probability cache: `runs/prob_cache/best-fc663d58e2ec7932`;
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

The 45-configuration refinement reused the same 60 cached probability maps and required no neural inference. It finished in 47:46.65 wall time with maximum RSS 1,023,800 kB, zero swaps, and exit status 0.

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

## Final 60-record bracket

Command:

```bash
/usr/bin/time -v python scripts/tune_postprocess.py \
    --data-dir data \
    --checkpoint runs/cpu_filanet_20epoch/best.pt \
    --limit 60 \
    --thresholds 0.91 0.92 0.925 0.93 0.935 0.94 \
    --merge-gap 40 \
    --min-area-fraction 0.00021 0.00023 0.00025 0.00027 0.00029 0.00031 \
    --min-confidence 0 \
    --out runs/cpu_filanet_20epoch/postprocess_trueval_60_finalbracket.json
```

The run reused all cached probability maps and swept 36 post-processing configurations in 34:55.79 wall time, with maximum RSS 1,004,752 kB, zero swaps, and exit status 0.

### Bracketed optimum

The exact `threshold=0.93`, `merge_gap=40` area curve is now bracketed:

| min-area fraction | matched Dice |
|---:|---:|
| 2.1e-4 | 0.3351 |
| **2.3e-4** | **0.3398** |
| 2.5e-4 | 0.3130 |
| 2.7e-4 | 0.3078 |
| 2.9e-4 | 0.2940 |
| 3.1e-4 | 0.2735 |

Thus `2.3e-4` is no longer a grid-boundary winner. The same bracket also shows that lowering the threshold does not produce a superior compensated operating point: the best values around threshold 0.91--0.925 remain below the exact 0.93 winner. The two intermediate thresholds were printed ambiguously in the old two-decimal table (`0.925` appeared as `0.93`, `0.935` as `0.94`); this display issue is now fixed in the tuner.

### Current candidate for full validation

| parameter / metric | value |
|---|---:|
| threshold | **0.93** |
| merge gap | **40.0** |
| min-area fraction | **2.3e-4** |
| matched Dice | **0.3398** |
| matched Dice over truth | 0.4756 |
| matched Dice over prediction | 0.4551 |
| mean paired Dice | 0.6421 |
| foreground Dice | 0.5265 |
| PQ | 0.2471 |
| RQ | 0.3804 |
| mean instances / annotation record | 6.7333 |
| spurious | 99 |
| one-to-many | 26 |
| many-to-one | 11 |
| missed | 151 |

The optimum is now sufficiently bracketed to stop tuning on the 60-record subset.

## Interpretation

1. Instance filtering is materially improving the competition-aligned proxy without changing the neural network.
2. The main gain comes from suppressing small false components: matched Dice improves while foreground Dice falls slightly, demonstrating that union-mask overlap and per-filament scoring prefer different operating points.
3. The refined mean prediction count of 6.73 instances per annotation record is much closer in magnitude to the MAGFiLO dataset-wide annotation density (~7.1 filaments per annotation record) than the original test submission (~11.9 predicted instances per image). This is supportive evidence, not a scoring target by itself.
4. Mean paired Dice remains almost unchanged (~0.642), while matched Dice improves from 0.313 to 0.340. Thus the improvement is almost entirely from better instance selection/counting rather than boundary quality.
5. The current best is still a tuning-subset result, not a final held-out estimate and not a Kaggle score.

## Full 168-record validation plan

The next experiment is no longer another parameter search. Run a very small candidate set across the complete grouped held-out split (168 annotation records / 106 physical images). Recommended candidates:

1. primary: threshold 0.93, merge gap 40, min-area fraction 2.3e-4;
2. slightly less aggressive filter: threshold 0.93, merge gap 40, min-area fraction 2.1e-4;
3. neighboring threshold: threshold 0.92, merge gap 40, min-area fraction 2.3e-4.

The shared physical-image cache should migrate/reuse the 57 physical images already present in the previous 60-record cache and infer only the remaining ~49 held-out images. The full-split winner, not the 60-record winner, determines Submission 2.
