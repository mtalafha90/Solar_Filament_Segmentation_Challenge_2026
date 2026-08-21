# Submission 2 pre-upload record — CPU E20 post-processing

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

## Status

**Pre-upload candidate.** Perform a round-trip challenge-CSV integrity check before Kaggle upload. Record the Kaggle public score only after the submission is accepted and scored.
