# Organizer-style metric audit

Date: 2026-08-22

## Why this was added

Submission 1 and Submission 2 used the same E20 checkpoint, while a large local
improvement in the custom matched-Dice proxy produced only a small Kaggle move
(0.20 -> 0.21). That is evidence that the previous optimization target was not
sufficiently aligned with the public grader.

A public competition implementation (`bcap52/solar-filament-segmentation-2026`)
ports the organizer Self_Evaluation_Notebook semantics as:

- score each annotator-image set separately against all predictions for the
  corresponding physical image;
- preserve GT instances as independent masks;
- hit when IoU is **strictly greater than 0.5**;
- TP is every hit pair; a prediction with no hit is FP; a GT instance with no
  hit is FN;
- form PQ only after accumulating TP IoU, FP and FN globally:

  `PQ = sum(TP IoU) / (TP + 0.5*FP + 0.5*FN)`

Relevant public reference:

- https://github.com/bcap52/solar-filament-segmentation-2026/blob/main/src/pq.py

This repository previously had a mathematically similar per-record PQ helper,
but tuning primarily optimized a matched-Dice proxy and GT instance masks could
be obtained from an integer label map. An integer label map overwrites pixels
where independent annotations overlap. The organizer-style path therefore uses
the original independent annotation masks and an IoU implementation that never
collapses them into labels.

## New code

- `src/filaseg/official_metric.py`
  - independent-mask pairwise IoU
  - strict IoU > 0.5 hit rule
  - streaming global TP/FP/FN accumulator
  - global PQ/SQ/RQ
  - overlap IoU/Dice diagnostics
- `scripts/tune_official_pq.py`
  - reconstructs the same grouped validation split from the checkpoint
  - reuses the existing per-physical-image probability cache
  - scores original independent GT masks
  - ranks post-processing configurations by organizer-style global PQ
- `tests/test_official_metric.py`
  - perfect-match regression
  - strict `>0.5` regression
  - global-vs-per-record aggregation regression
  - overlapping/multi-hit GT regression
  - empty GT/prediction accounting

## Immediate E20 rescore

After pulling `main`, run:

```bash
cd ~/Workspace/Solar_Filament_Segmentation_Challenge_2026
conda activate filaments

export PYTHONPATH="$PWD/src:$PYTHONPATH"
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

python -m pytest -q tests/test_official_metric.py tests/test_tuning.py

/usr/bin/time -v python scripts/tune_official_pq.py \
    --data-dir data \
    --checkpoint runs/cpu_filanet_20epoch/best.pt \
    --limit 0 \
    --thresholds 0.93 \
    --merge-gap 18 40 \
    --min-area-fraction 0.00012 0.00023 \
    --min-confidence 0 \
    --out runs/cpu_filanet_20epoch/official_pq_audit_e20.json
```

This is a 4-configuration cross-product. It deliberately contains the original
E20-style post-processing (`gap=18`, `area=1.2e-4`) and Submission 2
(`gap=40`, `area=2.3e-4`) plus the two crossed controls. Neural inference should
be reused from the existing tile-256, TTA-off cache.

## Decision rule

Do not choose the next model using the old matched-Dice proxy alone. Use
organizer-style global PQ as the primary local selection metric and keep SQ, RQ,
TP, FP, FN, matched Dice and foreground Dice as diagnostics.

Once the E20 audit establishes the new local baseline, the next controlled model
benchmark is a native-resolution 1024-crop U-Net with an ImageNet-pretrained
ResNet34 encoder and a simple BCE + Dice objective. Supplied spine metadata
should not be used; any skeleton/boundary/box supervision should be derived from
the segmentation masks.
