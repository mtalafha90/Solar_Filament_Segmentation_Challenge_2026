# CPU E20 Full Held-Out Post-processing Validation

**Date:** 2026-08-21 (Asia/Dubai)  
**Checkpoint:** `runs/cpu_filanet_20epoch/best.pt`  
**Purpose:** select the final post-processing operating point for Submission 2 using the complete grouped held-out validation split.

## Validation population

- Grouped split seed: 0
- Validation fraction: 0.15
- Annotation records: **168**
- Distinct physical images: **106**
- Inference tile: **256**
- TTA: **off**
- Device: **CPU**
- Network weights fixed at the original epoch-20 checkpoint

This run follows the corrected validation/tuning protocol documented in `docs/POSTPROCESS_TUNING_FIXES.md`.

## Finalist grid

The full split evaluated only the four finalists selected from the earlier 60-record true-validation tuning experiments:

| threshold | merge gap | min-area fraction | matched Dice | foreground Dice | PQ | RQ | mean instances | spurious | one-to-many | missed |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.92 | 40 | 2.1e-4 | 0.3257 | 0.5322 | 0.2429 | 0.3747 | 7.8 | 419 | 69 | 353 |
| 0.92 | 40 | 2.3e-4 | 0.3278 | 0.5290 | 0.2447 | 0.3761 | 7.2 | 354 | 64 | 380 |
| 0.93 | 40 | 2.1e-4 | 0.3260 | 0.5280 | 0.2442 | 0.3756 | 7.4 | 356 | 84 | 381 |
| **0.93** | **40** | **2.3e-4** | **0.3332** | **0.5272** | **0.2496** | **0.3831** | **6.9643** | **309** | **77** | **402** |

## Full-split winner

| parameter / metric | value |
|---|---:|
| threshold | **0.93** |
| min confidence | **0.0** |
| merge gap | **40.0** |
| min-area fraction | **2.3e-4** |
| matched Dice | **0.3332** |
| matched Dice over truth | 0.4646 |
| matched Dice over prediction | 0.4434 |
| mean paired Dice | 0.6410 |
| foreground Dice | 0.5272 |
| PQ | 0.2496 |
| RQ | 0.3831 |
| mean predicted instances / annotation record | 6.9643 |
| spurious | 309 |
| one-to-many | 77 |
| many-to-one | 33 |
| missed | 402 |

## Stability relative to the 60-record tuning subset

The 60-record tuning subset selected the same operating point:

- threshold 0.93
- merge gap 40
- min-area fraction 2.3e-4
- matched Dice 0.3398

On all 168 held-out annotation records, the same configuration gives matched Dice 0.3332. The absolute change is -0.0066 (about -1.9% relative). This is a small generalization drop and, more importantly, the identity of the winning configuration does not change.

The associated instance-count diagnostics are also similar in scale when normalized per annotation record:

| quantity | 60-record tuning subset | full 168-record split |
|---|---:|---:|
| matched Dice | 0.3398 | 0.3332 |
| mean predicted instances | 6.7333 | 6.9643 |
| spurious / record | 1.650 | 1.839 |
| missed / record | 2.517 | 2.393 |
| one-to-many / record | 0.433 | 0.458 |
| many-to-one / record | 0.183 | 0.196 |

The parameter therefore appears stable enough to use for the next hidden-test submission.

## Comparison with Baseline Submission 1 validation

The original baseline used threshold 0.93 with the old/default instance construction. On the same full held-out split it reported:

- foreground Dice 0.5391519
- PQ 0.2274469
- RQ 0.3500616
- spurious 818
- one-to-many 146
- many-to-one 20
- missed 199

The new full-split operating point reports:

- foreground Dice 0.5272
- PQ 0.2496
- RQ 0.3831
- spurious 309
- one-to-many 77
- many-to-one 33
- missed 402

The new post-processing deliberately sacrifices some union foreground overlap and recall to suppress small false/fragmented instances. Relative to the old full validation, spurious predictions fall by about 62% and one-to-many fragmentations fall by about 47%, while PQ and RQ improve by about 10%. Missed filaments increase substantially, so this is not a universal segmentation improvement; it is a controlled shift toward the competition-aligned instance operating point.

The mean paired Dice remains around 0.64, confirming that the network's matched masks are materially better than the final instance-level score. Detection/counting and unmatched structures remain the dominant bottleneck.

## Runtime

Full-finalist run:

- configurations: 4
- wall time: **52:25.02**
- user CPU time: **4450.60 s**
- system time: **918.73 s**
- CPU utilization: **170%**
- maximum resident set size: **1,347,116 kB**
- swaps: **0**
- exit status: **0**
- result JSON: `runs/cpu_filanet_20epoch/postprocess_full168_finalists.json`

## Submission 2 candidate

The validated inference/post-processing settings are:

```text
checkpoint          runs/cpu_filanet_20epoch/best.pt
threshold           0.93
min_confidence      0.0
merge_gap           40.0
min_area_fraction   0.00023
tile_size           256
TTA                 off
device              CPU
```

Recommended prediction command:

```bash
python scripts/predict.py --images data/test \
    --checkpoint runs/cpu_filanet_20epoch/best.pt \
    --out runs/cpu_filanet_20epoch/submission_epoch20_postproc2.csv \
    --format challenge \
    --threshold 0.93 \
    --min-confidence 0.0 \
    --merge-gap 40.0 \
    --min-area-fraction 0.00023 \
    --tile-size 256 \
    --device cpu \
    --no-tta
```

The generated CSV must still pass the existing challenge-format and RLE round-trip checks before upload. Kaggle public score must be recorded separately and must not be inferred from the internal matched-Dice proxy.
