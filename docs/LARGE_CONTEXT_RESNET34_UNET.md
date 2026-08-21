# Large-context ResNet34 U-Net experiment

## Why this replaces further E20 tuning

Submission 1 and Submission 2 used the same E20 checkpoint and scored 0.20 and
0.21 respectively despite a large reduction in predicted fragments. That makes
additional threshold/min-area/merge-gap tuning a low-priority path. The next
experiment changes the learned representation and the amount of physical
context seen during training.

## Controlled changes

This run intentionally changes only a few high-value factors:

1. **ImageNet-pretrained ResNet34 encoder** instead of the E20 scratch encoder.
2. **Native 1024x1024 crops** instead of 256x256 crops. The 2048-pixel GONG
   frame is not resized; each crop is taken at original resolution.
3. **Plain U-Net bottleneck**: no self-attention or edge-attention block, making
   1024-pixel crops practical and giving a clean architectural control.
4. **Mask-only BCE + Dice loss**: 0.5 BCE + 0.5 soft Dice, no clDice, focal,
   spine, boundary or deep-supervision losses, and no handcrafted distance
   weighting.
5. The existing **grouped train/validation split** remains unchanged so
   different annotations of the same physical GONG frame cannot leak across the
   split.

The model still receives the existing two input channels (photometrically
flattened H-alpha intensity and mu geometry). Extra annotation metadata may be
loaded by the dataset object for backwards-compatible caching, but this
benchmark's model/loss consumes only the segmentation mask target.

## Configuration

`configs/resnet34_unet_1024.yaml`

Key values:

```yaml
patch_size: 1024
batch_size: 1
samples_per_epoch: 1000
epochs: 24
learning_rate: 3e-4
positive_fraction: 0.70
selection_metric: pq

model:
  encoder: resnet34
  pretrained: true
  bottleneck_attention: false
  aux_heads: false
  deep_supervision: false

loss:
  bce: 0.5
  dice: 0.5
  tversky: 0.0
  cl_dice: 0.0
  focal: 0.0
  spine: 0.0
  boundary: 0.0
  deep: 0.0
  use_distance_weight: false
```

`model.depth: 6` in the YAML is not used by the ResNet encoder. It is present
only because the legacy geometry preflight uses that field to bound the
quadratic-attention token count; this benchmark has no bottleneck attention.

## Recommended execution

This is a GPU experiment. Do not spend days running the full 1024 configuration
on the old CPU laptop.

After pulling `main`, first run the regression tests:

```bash
python -m pytest -q tests/test_large_context_unet.py tests/test_losses.py tests/test_model.py
```

Then launch the benchmark on a CUDA machine/Kaggle GPU:

```bash
python scripts/train.py \
  --config configs/resnet34_unet_1024.yaml \
  --data-dir data \
  --device cuda
```

### One-epoch smoke run

Before the full run, verify memory and data flow without changing the scientific
configuration permanently:

```bash
python scripts/train.py \
  --config configs/resnet34_unet_1024.yaml \
  --data-dir data \
  --device cuda \
  --epochs 1 \
  --samples-per-epoch 64 \
  --val-max-images 4
```

Use `batch_size=1` first. If the GPU has substantial memory headroom, batch 2 can
be tested separately; do not change crop resolution merely to increase batch
size.

## Decision rule

The E20 post-processing work is now a frozen baseline. Advance the 1024 model
only if it improves held-out instance/PQ behaviour, not merely training loss or
foreground Dice. After training, evaluate the best checkpoint on the complete
grouped validation split and inspect PQ/RQ, matched-object Dice, missed and
spurious instances before creating a new Kaggle submission.

If this simple large-context pretrained U-Net produces a material gain, the next
model should be an instance-first detector/refiner using boxes derived from the
allowed segmentation masks. If it does not, the next investigation should focus
on instance formulation rather than adding more semantic-loss terms.
