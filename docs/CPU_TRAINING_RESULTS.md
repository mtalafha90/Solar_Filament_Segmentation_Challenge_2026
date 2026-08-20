# CPU MAGFiLO training results

This document records the real-data CPU training experiments performed on MAGFiLO for the Solar Filament Segmentation Challenge 2026. These are **internal validation results**, not Kaggle leaderboard scores. The current training validation uses whole-disk foreground overlap plus diagnostic instance metrics; the exact organizer instance-matching implementation has not yet been reproduced from public reference code.

## Hardware and software

- OS: Ubuntu 24.04.4 LTS (kernel 7.0.0-29-generic)
- CPU: Intel Core i7-5500U @ 2.40 GHz
- Physical cores / logical CPUs: 2 / 4
- Maximum CPU frequency: 3.0 GHz
- RAM: 11 GiB
- Swap: 4 GiB
- Root disk: 439 GiB, about 178 GiB free during setup
- Conda environment: `filaments`
- Python: 3.11.15
- PyTorch: 2.6.0+cu124, running CPU-only (`torch.cuda.is_available() == False`)
- CPU backend: MKL + oneDNN/MKL-DNN + OpenMP, AVX2/FMA available
- CPU thread settings used for the main experiments:
  - `OMP_NUM_THREADS=2`
  - `MKL_NUM_THREADS=2`
  - `OPENBLAS_NUM_THREADS=2`
  - `NUMEXPR_NUM_THREADS=2`

The CUDA-enabled PyTorch wheel was retained because it runs correctly in CPU fallback mode; reinstalling a CPU-only wheel was not required.

## Dataset audit

Dataset inspection was run with `scripts/inspect_data.py` on the distributed MAGFiLO data.

- Training image files: 707 JPEG images
- Test image files: 180
- Annotation records: 1154
- Distinct training frame names: 707
- Filament annotations: 8199
- Annotation records with spines: 8199
- Duplicate/independent annotator records: 447
- All 707 referenced frame names resolved successfully
- Missing training images: 0
- Sampled image size: 2048 x 2048
- Annotation/image size mismatches: 0
- Usable annotated observations in the training pipeline: 1154
- Sampled solar radius: 902.3 px
- Sampled filaments per image: 11.2
- Sampled filament disk coverage: mean 1.292%, min 0.722%, max 1.935%
- Suggested positive-class weight from the real data: `pos_weight = 8.8`
- Dataset-driven native-resolution patch recommendation: 512 px
- Suggested post-processing minimum area at this scale: 307 px

The repeated annotations are intentionally retained as separate observations, but train/validation splitting is grouped by physical frame so the same underlying image cannot appear on both sides of the split.

## CPU execution strategy

The repository default workload (`patch_size=512`, `batch_size=4`, `samples_per_epoch=2000`, `num_workers=4`) is intended for much stronger hardware and was not used on this laptop.

The CPU experiments retained the full FilaNet model architecture but used native-resolution crops with:

- `device: cpu`
- `patch_size: 256`
- `batch_size: 1` for training experiments
- `num_workers: 0`
- `amp: false`
- `pos_weight: 8.8`
- `selection_metric: dice`
- `val_tile: 256`
- conservative validation subsets during training

The 256 px setting crops the original 2048 px observations; it does **not** downsample the image, so native filament pixel structure is retained.

## Smoke test

Configuration:

- 1 epoch
- 32 training crops
- batch size 1
- 256 px patches
- 2 validation observations
- full FilaNet architecture

Hardware result:

- Training-only time: 64.4 s
- 2.0125 s per training crop
- Total wall time including validation/startup: 3 min 33.83 s
- CPU utilization: 170%
- Peak resident memory: 2,638,900 kB (~2.64 GB)
- Swap: 0
- Exit status: 0

Initial validation after only 32 training crops was intentionally not treated as scientifically meaningful. The selected Dice was 0.0357 at threshold 0.70, with precision 0.0190 and recall 0.6611.

### Initial loss components

| Loss component | Value |
|---|---:|
| BCE | 3.1641 |
| Tversky | 0.7965 |
| Focal | 0.2561 |
| clDice loss | 0.8920 |
| Spine | 0.3310 |
| Boundary | 0.8005 |
| Deep supervision | 3.6224 |
| Total | 6.1305 |

All components were finite; no NaNs, Infs, OOM failures, or swap activity occurred.

## Batch-size benchmark

A 32-crop benchmark compared batch sizes 1 and 2 at 256 px.

| Batch size | Seconds/sample | Samples/s |
|---:|---:|---:|
| 1 | 2.0125 | 0.497 |
| 2 | 1.9719 | 0.507 |

Batch size 2 was only about 2% faster per sample. Batch size 1 was retained for the longer CPU experiments because the throughput gain was negligible and batch 1 provides twice as many optimizer updates for the same number of sampled crops.

## Five-epoch pilot

Configuration:

- 5 epochs
- 200 crops/epoch
- 1000 crops total
- batch size 1
- 256 px patches
- `pos_weight=8.8`
- validation at epoch 5 on 4 held-out annotation records during the training run

### Training progression

| Epoch | Total loss | Training time (s) |
|---:|---:|---:|
| 1 | 3.8592 | 369.6 |
| 2 | 3.2341 | 822.6 |
| 3 | 3.4589 | 735.6 |
| 4 | 2.9797 | 624.8 |
| 5 | 2.9200 | 625.1 |

The loss decreased overall from 3.8592 to 2.9200. Variation in epoch time is consistent with data/cache access differences in addition to neural-network compute.

The training-run validation selected threshold 0.90 with:

- Dice: 0.3984
- IoU: 0.2530
- clDice: 0.4147
- Precision: 0.4084
- Recall: 0.4552
- PQ diagnostic: 0.1875

The threshold was at the upper edge of the initial grid, so a wider calibration was run before interpreting the optimum.

### Wider 12-observation threshold calibration

The same 5-epoch checkpoint was recalibrated on 12 fixed held-out observations using thresholds from 0.70 to 0.99.

| Threshold | Dice | IoU | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.70 | 0.3616 | 0.2303 | 0.2732 | 0.6933 |
| 0.75 | 0.3792 | 0.2438 | 0.2981 | 0.6683 |
| 0.80 | 0.3965 | 0.2571 | 0.3260 | 0.6397 |
| 0.85 | 0.4145 | 0.2709 | 0.3595 | 0.6060 |
| 0.88 | 0.4254 | 0.2793 | 0.3844 | 0.5815 |
| 0.90 | 0.4315 | 0.2840 | 0.4034 | 0.5617 |
| 0.92 | 0.4375 | 0.2886 | 0.4266 | 0.5393 |
| **0.94** | **0.4399** | **0.2903** | **0.4539** | **0.5088** |
| 0.96 | 0.4390 | 0.2891 | 0.4933 | 0.4674 |
| 0.97 | 0.4340 | 0.2846 | 0.5181 | 0.4365 |
| 0.98 | 0.4194 | 0.2719 | 0.5514 | 0.3882 |
| 0.99 | 0.3788 | 0.2387 | 0.6172 | 0.3034 |

For this 5-epoch checkpoint the calibrated optimum was therefore interior, at threshold 0.94:

- Dice: 0.4399
- IoU: 0.2903
- Precision: 0.4539
- Recall: 0.5088
- clDice: 0.4540
- PQ diagnostic: 0.1651

## Twenty-epoch CPU run

Configuration:

- 20 epochs
- 200 crops/epoch
- 4000 training crops total
- batch size 1
- patch size 256
- full FilaNet architecture
- `pos_weight=8.8`
- `selection_metric=dice`
- validation every 5 epochs
- validation subset capped at 12 annotation records during training
- threshold grid extended through 0.99

Total execution statistics:

- Wall time: 2 h 58 min 16 s
- User CPU time: 17,996.23 s
- System time: 815.04 s
- Average CPU utilization: 175%
- Peak resident memory: 3,109,176 kB (~3.11 GB)
- Swap: 0
- Exit status: 0

### Learning curve

| Epoch | Time (s) | Train loss | Val Dice | Val IoU | Precision | Recall | Best threshold |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 355.9 | 3.6304 | - | - | - | - | - |
| 2 | 348.4 | 3.2534 | - | - | - | - | - |
| 3 | 349.0 | 3.8836 | - | - | - | - | - |
| 4 | 353.4 | 2.9941 | - | - | - | - | - |
| 5 | 352.3 | 3.1460 | 0.4270 | 0.2829 | 0.4588 | 0.4854 | 0.94 |
| 6 | 527.8 | 3.6646 | - | - | - | - | - |
| 7 | 626.2 | 2.9674 | - | - | - | - | - |
| 8 | 566.5 | 2.5118 | - | - | - | - | - |
| 9 | 458.6 | 2.4139 | - | - | - | - | - |
| 10 | 471.9 | 2.6388 | 0.3941 | 0.2553 | 0.3968 | 0.4848 | 0.99 |
| 11 | 424.0 | 2.4412 | - | - | - | - | - |
| 12 | 426.4 | 1.9305 | - | - | - | - | - |
| 13 | 381.3 | 2.6275 | - | - | - | - | - |
| 14 | 397.4 | 2.3591 | - | - | - | - | - |
| 15 | 388.6 | 2.2822 | 0.4828 | 0.3296 | 0.4864 | 0.5949 | 0.94 |
| 16 | 396.3 | 2.1721 | - | - | - | - | - |
| 17 | 367.3 | 2.2907 | - | - | - | - | - |
| 18 | 358.9 | 2.3077 | - | - | - | - | - |
| 19 | 369.2 | 1.9350 | - | - | - | - | - |
| **20** | **363.8** | **1.9933** | **0.5366** | **0.3730** | **0.5476** | **0.5888** | **0.95** |

The best checkpoint was epoch 20 (`best.pt` and `last.pt` both correspond to epoch 20). Training loss continued to decline overall while the best measured validation Dice occurred at the final epoch, so this run does not show a clear generalization plateau by epoch 20. The epoch-10 decline is treated as validation-subset noise rather than evidence of persistent overfitting, because performance recovered substantially at epochs 15 and 20.

### Epoch-20 threshold curve

| Threshold | Dice | IoU | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.50 | 0.3503 | 0.2190 | 0.2305 | 0.8725 |
| 0.60 | 0.3880 | 0.2486 | 0.2666 | 0.8515 |
| 0.70 | 0.4275 | 0.2807 | 0.3096 | 0.8217 |
| 0.75 | 0.4488 | 0.2984 | 0.3363 | 0.8010 |
| 0.80 | 0.4722 | 0.3182 | 0.3691 | 0.7748 |
| 0.85 | 0.4964 | 0.3390 | 0.4098 | 0.7392 |
| 0.88 | 0.5102 | 0.3508 | 0.4390 | 0.7100 |
| 0.90 | 0.5187 | 0.3579 | 0.4621 | 0.6840 |
| 0.92 | 0.5269 | 0.3648 | 0.4897 | 0.6540 |
| 0.94 | 0.5346 | 0.3712 | 0.5253 | 0.6148 |
| **0.95** | **0.5366** | **0.3730** | **0.5476** | **0.5888** |
| 0.96 | 0.5347 | 0.3712 | 0.5734 | 0.5559 |
| 0.97 | 0.5290 | 0.3663 | 0.6077 | 0.5138 |
| 0.98 | 0.5117 | 0.3514 | 0.6570 | 0.4532 |
| 0.99 | 0.4371 | 0.2868 | 0.7304 | 0.3330 |

The threshold optimum is now clearly interior at 0.95, so the earlier boundary-calibration concern is resolved for this checkpoint.

At threshold 0.95 the 12-observation training-time validation summary was:

- Dice: 0.5366
- IoU: 0.3730
- clDice: 0.5847
- MSIoU: 0.3122
- Precision: 0.5476
- Recall: 0.5888
- PQ diagnostic: 0.2102
- SQ diagnostic: 0.5882
- RQ diagnostic: 0.3287
- one-to-many count: 14
- many-to-one count: 0
- missed count: 16
- spurious count: 70

The instance diagnostics suggest that, on this small validation subset, the current weakness is more associated with fragmentation/spurious components than with over-merging. These counts are diagnostic and must be re-evaluated on the complete held-out split before tuning post-processing around them.

## Current interpretation

1. Full FilaNet is technically viable on the i7-5500U CPU at 256 px native-resolution crops.
2. Memory is not the limiting factor: the 20-epoch run peaked at only ~3.11 GB resident memory and used no swap.
3. Compute time is the limiting factor. The repository's full GPU-oriented default schedule would require several days on this CPU, so CPU experiments use smaller epoch sample budgets and less frequent validation.
4. Real MAGFiLO learning is clearly present: the internal validation Dice increased from 0.0357 after the 32-crop smoke test, to 0.4399 for the calibrated 5-epoch checkpoint, to 0.5366 at epoch 20 on the 12-observation training validation subset.
5. The 20-epoch learning curve does not yet show a convincing plateau; however, a longer run should not be launched until the best checkpoint is scored over the complete held-out split.
6. The current Dice values are internal foreground-overlap validation values. They must not be reported as Kaggle leaderboard scores or as the exact organizer mean-Dice metric until the organizer matching/scoring implementation is verified.

## Full held-out validation

**Status: running / result pending.**

The next evaluation reconstructs the same grouped split and scores `runs/cpu_filanet_20epoch/best.pt` on all held-out annotation records rather than the 12-record training-time cap. The planned threshold range is 0.90-0.98 around the calibrated optimum. The result from that run should be added here before deciding whether to continue training to 30-40 epochs or instead tune the loss/post-processing/instance behavior.
