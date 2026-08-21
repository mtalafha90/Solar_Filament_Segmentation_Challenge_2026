# Submission 2 result — CPU E20 post-processing

**Date:** 2026-08-22 (Asia/Dubai)

## Candidate

This candidate keeps the CPU E20 network weights fixed and changes only instance post-processing, selected on the complete grouped held-out split.

- checkpoint: `runs/cpu_filanet_20epoch/best.pt`
- threshold: `0.93`
- merge gap: `40.0`
- min-area fraction: `0.00023`
- min confidence: `0.0`
- tile size: `256`
- TTA: off
- device: CPU

## Full grouped held-out validation

Validation set: 168 annotation records / 106 physical images.

| metric | value |
|---|---:|
| matched Dice proxy | 0.3332 |
| matched Dice / truth | 0.4646 |
| matched Dice / prediction | 0.4434 |
| mean paired Dice | 0.6410 |
| foreground Dice | 0.5272 |
| PQ | 0.2496 |
| RQ | 0.3831 |
| mean predicted instances / record | 6.9643 |
| spurious | 309 |
| one-to-many | 77 |
| many-to-one | 33 |
| missed | 402 |

The same parameter setting won on the 60-record tuning subset (`matched_dice=0.3398`) and on the complete 168-record grouped validation split (`matched_dice=0.3332`), indicating that the selected operating point is stable rather than an isolated subset optimum.

## Test prediction generation

Command:

```bash
/usr/bin/time -v python scripts/predict.py \
    --images data/test \
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

Runtime and resource usage:

- 180 test images
- wall time: 2:36:09
- user CPU: 14,348.07 s
- system CPU: 3,039.53 s
- CPU utilization: 185%
- peak RSS: 5,337,920 kB
- swap: 0
- exit status: 0

## Test prediction summary

| quantity | value |
|---|---:|
| submission rows / total instances | 1233 |
| images | 180 |
| mean instances / image | 6.85 |
| maximum instances / image | 18 |
| images with no detection | 2 |
| mean pixel coverage | 0.0033 |
| median instance area | 1155 px |
| minimum instance area | 541 px |
| maximum instance area | 69,315 px |

The test prediction density (6.85 instances/image) closely matches the full-validation prediction density (6.96 instances/record) and is far below Baseline Submission 1 (2142 rows / 180 images = 11.9 instances/image). This is consistent with the intended reduction in small spurious/fragmented detections. Count agreement is a sanity check, not a scoring target.

## Final round-trip integrity check

The generated challenge CSV was decoded row-by-row before upload.

```text
SUBMISSION 2 INTEGRITY CHECK
==================================================
CSV rows                 : 1233
Expected test images     : 180
Images represented       : 178
Images with no prediction: 2
Duplicate filament IDs   : 0
Unexpected image IDs     : 0
Bad filament numbering   : 0
Empty decoded masks      : 0
Bad decoded shapes       : 0
Minimum decoded area     : 541
Maximum decoded area     : 69315
No-detection images:
   20200310113230Ch
   20210811072510Th

SUBMISSION 2 ROUND-TRIP CHECK: PASS
```

The two absent image IDs are intentional no-detection cases; challenge-format output contains no empty row for an image with zero predicted filaments. All 1233 encoded masks decoded to non-empty 2048×2048 masks, with unique and consecutively numbered filament IDs and no unexpected image IDs.

## Kaggle result

**Public leaderboard score: 0.21** (user-reported Kaggle display precision).

For comparison:

| submission | network | test instances | instances/image | public score |
|---|---|---:|---:|---:|
| Baseline Submission 1 | CPU E20 `best.pt` | 2142 | 11.90 | 0.20 |
| Submission 2 | same CPU E20 `best.pt` | 1233 | 6.85 | **0.21** |

The only material change between the two submissions was instance post-processing. Therefore the `0.20 -> 0.21` public-score increase shows that aggressive suppression/merging of small components helped only marginally on the current public grader. This is much smaller than the improvement in the local matched-Dice proxy and establishes that the current local proxy is not a sufficiently accurate surrogate for the Kaggle public metric.

This result also argues against further broad tuning of threshold, merge gap and minimum area on the same checkpoint. Future work should prioritize (1) understanding/replicating the public scoring implementation as far as the organizers disclose it, and (2) improving the learned representation and instance formulation rather than spending CPU on additional morphology-only sweeps.

## Status

**SUBMITTED AND SCORED.**

Upload artifact:

```text
runs/cpu_filanet_20epoch/submission_epoch20_postproc2.csv
```

Kaggle label:

```text
FilaNet CPU E20 | instance-tuned | thr0.93 | area2.3e-4 | gap40
```
