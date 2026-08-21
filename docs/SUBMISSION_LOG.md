# Competition Submission Log

This document records Kaggle submission artifacts and the exact experiment state used to generate them for the **Solar Filament Segmentation Challenge 2026**. It is intended to make every leaderboard submission reproducible and traceable back to its checkpoint, validation result, inference settings, and local artifacts.

> **Important:** internal validation scores in this repository are not claimed to be Kaggle leaderboard scores. The leaderboard score for each submission is recorded separately when Kaggle returns it.

---

## Baseline Submission 1 — FilaNet CPU E20

**Status:** submission CSV generated and verified; Kaggle upload / leaderboard score pending  
**Recorded:** 2026-08-21 (Asia/Dubai)  
**Submission label:** `FilaNet CPU E20 | patch256 | thr0.93 | no-TTA`

### Model provenance

- Model: full FilaNet
- Training device: CPU
- Checkpoint: `runs/cpu_filanet_20epoch/best.pt`
- Best checkpoint epoch: **20**
- `last.pt` epoch: **20**
- Patch size: **256 px**, native-resolution crops from 2048 x 2048 frames
- Batch size: **1**
- Samples per epoch: **200**
- Total epochs: **20**
- Total sampled training crops: **4000**
- Positive sampling fraction: **0.7**
- `pos_weight`: **8.8**
- Learning rate: **3e-4**
- Weight decay: **1e-4**
- Warm-up: **2 epochs**
- Mixed precision: off
- DataLoader workers: **0**
- Selection metric: **Dice**
- Validation during training: every 5 epochs, capped at 12 held-out annotation records
- CPU thread settings:
  - `OMP_NUM_THREADS=2`
  - `MKL_NUM_THREADS=2`
  - `OPENBLAS_NUM_THREADS=2`
  - `NUMEXPR_NUM_THREADS=2`

The complete CPU training history, dataset audit, smoke tests, and learning curve are recorded in `docs/CPU_TRAINING_RESULTS.md`.

### Twenty-epoch learning checkpoints

| Epoch | Train loss | Val Dice | Val IoU | Precision | Recall | Best threshold |
|---:|---:|---:|---:|---:|---:|---:|
| 5 | 3.1460 | 0.4270 | 0.2829 | 0.4588 | 0.4854 | 0.94 |
| 10 | 2.6388 | 0.3941 | 0.2553 | 0.3968 | 0.4848 | 0.99 |
| 15 | 2.2822 | 0.4828 | 0.3296 | 0.4864 | 0.5949 | 0.94 |
| **20** | **1.9933** | **0.5366** | **0.3730** | **0.5476** | **0.5888** | **0.95** |

The epoch-20 checkpoint was the best checkpoint from the training run.

### Full held-out validation used for submission threshold

The epoch-20 checkpoint was re-evaluated on the complete grouped held-out split:

- Held-out annotation records: **168**
- Distinct physical frames represented: **106**
- Best full-validation threshold: **0.93**

| Threshold | Dice | IoU | Precision | Recall |
|---:|---:|---:|---:|---:|
| 0.90 | 0.5354 | 0.3819 | 0.5086 | 0.6426 |
| 0.92 | 0.5387 | 0.3849 | 0.5347 | 0.6146 |
| **0.93** | **0.5392** | **0.3853** | **0.5497** | **0.5976** |
| 0.94 | 0.5381 | 0.3841 | 0.5663 | 0.5777 |
| 0.95 | 0.5347 | 0.3806 | 0.5843 | 0.5536 |
| 0.96 | 0.5276 | 0.3735 | 0.6054 | 0.5233 |
| 0.97 | 0.5141 | 0.3606 | 0.6306 | 0.4836 |
| 0.98 | 0.4869 | 0.3351 | 0.6646 | 0.4255 |

Full held-out summary at threshold 0.93:

- Dice: **0.5391518818126046**
- IoU: **0.38530396558643143**
- Precision: **0.5497351128527206**
- Recall: **0.5976305666089161**
- clDice: **0.6072773554993222**
- MSIoU: **0.3341070407270626**
- PQ diagnostic: **0.22744687977865516**
- SQ diagnostic: **0.5764129602282472**
- RQ diagnostic: **0.35006162688581743**
- one-to-many: **146**
- many-to-one: **20**
- missed: **199**
- spurious: **818**

The full validation moved the operating threshold from the checkpoint-stored 0.95 to **0.93**. Therefore the submission explicitly overrides the checkpoint threshold with `--threshold 0.93`.

### Full held-out validation runtime

- Wall time: **2 h 34 min 24 s**
- Evaluation elapsed time reported by the script: **154.3 min**
- User CPU time: **14,073.17 s**
- System time: **2,664.70 s**
- CPU utilization: **180%**
- Peak resident memory: **3,232,564 kB (~3.23 GB)**
- Swap: **0**
- Exit status: **0**
- Saved local summary: `runs/cpu_filanet_20epoch/full_val_summary.json`

### Test inference configuration

Submission inference intentionally matches the validated inference procedure:

- Test directory: `data/test`
- Number of test images: **180**
- Checkpoint: `runs/cpu_filanet_20epoch/best.pt`
- Probability threshold: **0.93**
- Tile size: **256**
- Device: CPU
- Test-time augmentation: **disabled**
- Output format: challenge CSV

Exact command:

```bash
cd ~/Workspace/Solar_Filament_Segmentation_Challenge_2026
conda activate filaments

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2
export PYTHONPATH="$PWD/src:$PYTHONPATH"

/usr/bin/time -v python scripts/predict.py \
    --images data/test \
    --checkpoint runs/cpu_filanet_20epoch/best.pt \
    --out runs/cpu_filanet_20epoch/submission_epoch20_thr093.csv \
    --format challenge \
    --threshold 0.93 \
    --tile-size 256 \
    --device cpu \
    --no-tta
```

### Generated submission artifact

Local file:

```text
runs/cpu_filanet_20epoch/submission_epoch20_thr093.csv
```

Absolute local path used for manual Kaggle upload:

```text
/home/talafha/Workspace/Solar_Filament_Segmentation_Challenge_2026/runs/cpu_filanet_20epoch/submission_epoch20_thr093.csv
```

Challenge CSV columns:

```text
filament_id,segmentation_rle
```

The repository writer emits one row per predicted filament, using compressed `pycocotools` RLE counts for a fixed 2048 x 2048 mask.

### Submission sanity checks

The generated CSV passed the repository-side format and decode checks.

Confirmed:

- CSV structure check: **PASS**
- RLE decode check: **PASS**
- Test images represented after decoding: **180**
- Total decoded predicted instances: **2142**
- Mean predicted instances per test image: **2142 / 180 = 11.9**
- All decoded masks have shape: **2048 x 2048**
- Duplicate filament IDs: **0**
- Empty RLE rows: **0**

The predicted instance count is of the same order as the sampled MAGFiLO training annotations (~11.2 filaments per image), so the submission does not show an obvious catastrophic count failure. This does not imply that every predicted instance is correct; full validation still shows substantial fragmentation and spurious-component errors.

First decoded test examples reported during the check:

| Test image | Predicted instances | Decoded shape |
|---|---:|---|
| `20110120105534Ch` | 2 | 2048 x 2048 |
| `20110130110334Ch` | 3 | 2048 x 2048 |
| `20110214175414Mh` | 15 | 2048 x 2048 |

First five CSV rows inspected:

| `filament_id` | RLE character count |
|---|---:|
| `20110120105534Ch_1` | 189 |
| `20110120105534Ch_2` | 185 |
| `20110130110334Ch_1` | 66 |
| `20110130110334Ch_2` | 71 |
| `20110130110334Ch_3` | 392 |

### Exact verification outcomes

```text
SUBMISSION FORMAT CHECK: PASS
Images represented: 180
Decoded instances: 2142
20110120105534Ch instances = 2 shape = (2048, 2048)
20110130110334Ch instances = 3 shape = (2048, 2048)
20110214175414Mh instances = 15 shape = (2048, 2048)
RLE DECODE CHECK: PASS
```

### Related local artifacts

These files are part of the scientific/reproducibility record but are **not** uploaded as the Kaggle prediction CSV:

| Artifact | Purpose | Kaggle prediction upload? |
|---|---|---|
| `runs/cpu_filanet_20epoch/best.pt` | Best trained model checkpoint | No |
| `runs/cpu_filanet_20epoch/last.pt` | Last epoch checkpoint | No |
| `runs/cpu_filanet_20epoch/history.json` | Epoch-by-epoch training/validation history | No |
| `runs/cpu_filanet_20epoch/full_val_summary.json` | Complete held-out validation summary | No |
| `runs/cpu_filanet_20epoch/submission_epoch20_thr093.csv` | Hidden-test predictions in challenge format | **Yes** |

The checkpoint and JSON files are retained locally because they are needed for reproducibility, later threshold/post-processing experiments, regeneration of predictions, and the final technical report. The competition prediction uploader needs the generated CSV rather than the model-training state.

### Leaderboard record

| Field | Value |
|---|---|
| Submission | Baseline Submission 1 |
| Description | `FilaNet CPU E20 | patch256 | thr0.93 | no-TTA` |
| CSV | `submission_epoch20_thr093.csv` |
| Checkpoint | `best.pt`, epoch 20 |
| Threshold | 0.93 |
| TTA | off |
| Internal full-held-out Dice | 0.5391518818 |
| Kaggle public score | **PENDING** |
| Kaggle submission status | **PENDING** |

When Kaggle returns the score, update the final two fields rather than replacing the internal validation result. This preserves the distinction between local scientific validation and external competition evaluation.

### Interpretation before leaderboard submission

This submission is deliberately a **baseline**, not a final model. The full held-out evaluation shows that the semantic segmentation is learning meaningfully, but instance construction remains the major weakness:

- spurious predictions: 818
- one-to-many fragmentations: 146
- many-to-one mergers: 20
- missed filaments: 199
- SQ (0.5764) is substantially higher than RQ (0.3501)

The next optimization work should therefore test post-processing / instance-construction changes on the fixed epoch-20 checkpoint before assuming that longer training alone is the best use of CPU time. Future submissions should receive their own numbered section in this log so that leaderboard changes can be attributed to one controlled modification at a time.
