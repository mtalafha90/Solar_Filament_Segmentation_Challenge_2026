# Results

Every number here comes from the **synthetic** generator in
`filaseg.data.synthetic`, because the competition data was not available in the
environment where this work was done. They are not competition scores. Their
purpose is to show that the pipeline is correct end to end and that each
component earns its place.

Real GONG observations are harder in ways synthetic data cannot capture — above
all the irregularity of genuine sunspot groups, the variability of seeing across
the six network sites, and filaments whose faint ends genuinely are ambiguous to
human annotators. Expect lower absolute numbers on MAGFiLO. The *relative*
findings below should transfer; verify them with `scripts/ablation.py` and
`scripts/tune_classical.py` once you have the data.

## Setup

- 40 synthetic full-disk observations at 384×384, solar radius ≈ 169 px,
  8 filaments and 5 sunspots per frame, filaments covering ≈ 3.7% of the disk.
- Split 85/15 into training and validation by `filaseg.train.split_ids(seed=0)`,
  giving 6 validation frames.
- FilaNet at `base_width=24, depth=3` (a small configuration, chosen so the
  ablation could run on CPU), 128-pixel patches, 240 samples per epoch,
  14 epochs. The default configuration in `configs/default.yaml` is larger.
- All metrics computed after full instance extraction, with eight-fold dihedral
  test-time augmentation, on whole disks.

Reproduce with:

```bash
python scripts/make_synthetic_dataset.py --out data/synthetic \
    --n-images 40 --size 384 --seed 1000
python scripts/ablation.py --annotations data/synthetic/annotations.json \
    --image-dir data/synthetic/images --epochs 14 --patch-size 128 \
    --base-width 24 --depth 3
```

## FilaNet against the training-free detector

Both scored on the same 6 validation frames, same ground truth, same metric code.

| Metric | Classical | FilaNet | Difference |
|---|---|---|---|
| IoU | 0.6215 | **0.8777** | +0.2562 |
| Dice | 0.7227 | **0.9347** | +0.2120 |
| Precision | **0.9400** | 0.9243 | −0.0157 |
| Recall | 0.6427 | **0.9457** | +0.3031 |
| clDice | 0.7468 | **0.9773** | +0.2306 |
| MSIoU | 0.6245 | **0.8630** | +0.2384 |
| Hit rate | 0.5833 | **0.9583** | +0.3750 |
| Pairwise IoU | 0.5181 | **0.8631** | +0.3451 |
| AP@0.50 | 0.5667 | **0.9542** | +0.3875 |
| AP@0.75 | 0.4372 | **0.8768** | +0.4396 |
| mAP | 0.5360 | **0.9284** | +0.3924 |

The classical detector is the more *precise* of the two — when it commits to a
filament it is usually right — but it misses far more, and the gap widens as the
IoU threshold rises. That is the expected shape of the comparison: hysteresis
thresholding recovers a filament's bright core reliably and its faint extremities
unreliably, so the boundaries are loose, which costs most at AP@0.75.

## What each component of the network contributes

| Variant | IoU | Dice | clDice | MSIoU |
|---|---|---|---|---|
| Full model | 0.8608 | 0.9251 | **0.9712** | 0.8238 |
| No clDice loss | **0.8836** | **0.9381** | 0.9601 | **0.8511** |
| No edge attention | 0.8601 | 0.9247 | 0.9689 | 0.8205 |
| No auxiliary heads | 0.8571 | 0.9229 | 0.9710 | 0.8213 |
| No deep supervision | 0.8653 | 0.9277 | 0.9682 | 0.8231 |

(Pixel-level validation scores during training, before instance extraction, which
is why they differ from the table above.)

Only one component produces a clearly directional effect at this scale. The
auxiliary heads and deep supervision move IoU by −0.004 and +0.005, which is
inside run-to-run variation on six validation frames.

**The clDice term does exactly what it is designed to do, and it is a trade.**
Removing it *raises* pixel IoU by 0.023 and *lowers* clDice by 0.011. It spends
area agreement to buy topological fidelity. Whether that is worth it depends on
what is being scored: for the challenge's fine-structure criterion it is, and
clDice and MSIoU are the metrics that see it. If you are optimising pixel IoU
alone, turn it down.

**Edge attention, the auxiliary heads and deep supervision show no measurable
effect on synthetic data.** Edge attention scores 0.8601 against the full
model's 0.8608. These are reported as null results rather than quietly omitted.
They are not evidence that the mechanisms fail — they are evidence that this
benchmark cannot resolve them. Synthetic barbs come from a simple generative
model with clean edges, so there is little for an edge prior to add; the
ablation model is deliberately small so it trains on a CPU; and six validation
frames is a thin basis for separating differences of half a per cent.

Treat all three as unproven and re-run the ablation on MAGFiLO, which is one
command. If they remain flat on real data, drop them: the full model is 9.7M
parameters and roughly 0.4M of that is the edge pathway.

## Post-processing

Applied to FilaNet's output on the validation frames:

| Configuration | IoU | Recall | Precision | clDice | Hit rate | Predictions |
|---|---|---|---|---|---|---|
| Threshold only | 0.8655 | 0.9441 | 0.9123 | 0.9724 | 0.958 | 75 |
| + small-object removal | **0.8777** | 0.9457 | 0.9243 | **0.9773** | 0.958 | 59 |
| + fragment merging | **0.8777** | 0.9457 | 0.9243 | **0.9773** | 0.958 | 57 |
| + sunspot rejection | 0.7330 | 0.7523 | **0.9651** | 0.8490 | 0.729 | 35 |

**Sunspot rejection must not be applied to a trained network's output.** It cost
0.145 IoU and a quarter of the hit rate. The reason is straightforward once
measured: the network never predicts sunspots in the first place, so the shape
filter has no sunspots to remove and deletes genuine short, compact filaments
instead. It is now off by default and the classical detector, which does need it,
enables it explicitly.

Small-object removal and fragment merging both help or are neutral. Merging
reduced 59 components to 57 without changing any metric on these frames — the
synthetic filaments are rarely broken. It matters more on real data, where poor
seeing genuinely fragments filaments.

> The synthetic figures in this document predate the defaults being retuned for
> real GONG statistics. Synthetic frames carry 3–7% filament coverage against
> MAGFiLO's 0.84%, so they were produced with `expected_coverage` around 0.012;
> the shipped default is now 0.004. On sparse GONG-scale frames the retuned
> defaults cut the false discovery rate from 0.63 to 0.06 and raised IoU from
> 0.651 to 0.740.

## The classical detector's coverage prior

The one knob that has to be set from data. Measured on a JPEG dataset whose
filaments cover 3.7% of the disk:

| `expected_coverage` | IoU | Recall | Precision | clDice | MSIoU |
|---|---|---|---|---|---|
| 0.012 (default) | **0.6104** | 0.6338 | **0.9383** | **0.7920** | **0.6451** |
| 0.0222 | 0.6037 | **0.7355** | 0.7249 | 0.7194 | 0.5967 |
| 0.0300 | 0.4321 | 0.7139 | 0.4896 | 0.5167 | 0.4328 |

Raising it buys recall and costs precision, and the net effect on IoU is not
predictable from coverage alone — which is why `scripts/inspect_data.py`
recommends a *range* to search rather than a single value, and points at
`scripts/tune_classical.py`.

## Component-level checks

Measured directly rather than through a trained model.

**Limb fitting.** Radius recovered to 0.07% and disk centre to 0.012 pixels,
averaged over six frames.

**Photometric flattening.** Filament-to-quiet-Sun contrast in three radial
bands: 0.537, 0.531, 0.549 — a spread of 3%, against tens of per cent in the raw
image. This is what makes a single threshold mean the same thing everywhere.

**Loss sensitivity to a deleted barb.** On a filament whose barb is 13.5% of its
pixels, deleting it raises the clDice loss by 37× and the Tversky loss by 7.7×.

**Metric behaviour.**

| Prediction | IoU | MSIoU | clDice |
|---|---|---|---|
| Exact | 1.000 | 1.000 | 1.000 |
| Shifted one pixel | 0.628 | 0.749 | 0.994 |
| Barb deleted | 0.914 | 0.908 | 0.940 |

**Sunspot rejection, classical detector, heavy sunspot load.** False discovery
rate 0.599 without the shape filter, 0.143 with it.

**Fragment merging.** Two collinear fragments separated by a 12-pixel gap are
merged; two parallel filaments 6 pixels apart are not; two collinear fragments
60 pixels apart are not.
