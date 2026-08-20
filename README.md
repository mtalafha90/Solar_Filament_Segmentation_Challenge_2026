# Solar Filament Segmentation Challenge 2026

Pixel-precise segmentation of solar filaments in full-disk H-alpha observations,
for the [IEEE Big Data Cup 2026 challenge](https://bigdataieee.org/BigData2026/cup/solar-filament-segmentation/)
hosted [on Kaggle](https://www.kaggle.com/competitions/filament-segmentation-2026)
and built around the [MAGFiLO](https://www.nature.com/articles/s41597-024-03876-y)
dataset of 10,244 manually annotated filaments in 1,593 GONG observations.

The repository contains two complete detectors and everything needed to train,
run and score them:

| | What it is | Needs training? |
|---|---|---|
| **FilaNet** | An edge-guided, multi-task U-Net | Yes |
| **Classical detector** | Ridge filtering with hysteresis thresholding | No |

> **Use FilaNet for real data.** Measured on GONG-like frames, the classical
> detector cannot exceed **IoU ≈ 0.08 at any threshold** — filament contrast
> against the chromospheric network is around 1σ, far too low for a hand-built
> score. It is worth running once as a pipeline check and as a fallback on
> observations unlike anything in training, but it is not a competitive
> baseline. See [Results](#results).

---

## Why this problem is not ordinary segmentation

Three properties of filaments shape every design decision here.

**They are thin, and the thin parts are what count.** A filament is a long dark
thread with short side-threads called *barbs*. The barbs are only a few pixels
across, so they make up a tiny share of the pixel count but carry most of the
scientific information — barbs reveal the filament's magnetic chirality. A model
trained on Dice loss alone learns to draw a smooth blob over each filament body
and delete every barb, because doing so costs it almost nothing.

**They are rare.** Filaments cover well under one per cent of the solar disk.
Sampled uniformly, almost every training patch would be empty quiet Sun.

**They are not the only dark things on the Sun.** Sunspots are darker than most
filaments and are the main source of false positives. Any detector keyed on
brightness alone finds sunspots first.

The pipeline addresses each of these directly, and the sections below say how.

---

> **New here? [`QUICKSTART.md`](QUICKSTART.md) has the exact commands**, from
> cloning to a submission file, including where to put the dataset.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

PyTorch is only needed for FilaNet. The classical detector, the metrics and the
whole data layer run on NumPy, SciPy and scikit-image alone.

## Quick start, with no data to hand

The repository ships a generator that produces synthetic full-disk observations
with limb darkening, transmission gradients, noise, sunspots and filaments with
barbs, laid out exactly like MAGFiLO. That means the whole pipeline can be run
and tested before the competition data has been downloaded:

```bash
# Build a small synthetic dataset
python scripts/make_synthetic_dataset.py --out data/synthetic --n-images 40

# Score the training-free detector on it
python scripts/evaluate.py \
    --annotations data/synthetic/annotations.json \
    --image-dir data/synthetic/images \
    --classical

# Train FilaNet on it
python scripts/train.py \
    --annotations data/synthetic/annotations.json \
    --image-dir data/synthetic/images \
    --cache-dir data/cache --output-dir runs/demo \
    --epochs 20 --patch-size 128 --batch-size 8
```

## Running on the competition data

The expected layout is the usual competition one — JPEGs in `train/` and
`test/`, with a COCO-style annotation JSON alongside the training images:

```
data/
  train/    lots of .jpg  +  annotations.json
  test/     lots of .jpg
```

**Start by inspecting it.** This verifies that every annotation resolves to a
readable image, checks that images and annotations agree on size, and measures
the statistics that decide how to configure both detectors:

```bash
python scripts/inspect_data.py --data-dir data
```

It finds `train/`, `test/` and the JSON on its own, and prints filament counts,
chirality balance, solar radius, the fraction of the disk covered by filaments,
and the settings those imply. Pass `--annotations` and `--image-dir` explicitly
if your layout differs.

Then train, score and predict:

```bash
# Train
python scripts/train.py --config configs/default.yaml --data-dir data \
    --cache-dir data/cache --output-dir runs/filanet

# Score on the held-out split
python scripts/evaluate.py --data-dir data --checkpoint runs/filanet/best.pt

# Predict on test/ and write the competition submission
python scripts/predict.py --images data/test \
    --checkpoint runs/filanet/best.pt --out submission.csv
```

The default output is the competition's format: one row per predicted filament,
keyed `<image_id>_<n>`, with the mask as pycocotools RLE counts and the size
omitted (every frame is 2048×2048). Images with no detection contribute no rows,
which is right — the grader matches by overlap, so a blank row would count as a
spurious segment. `read_challenge_csv()` decodes a submission back to masks if
you want to check one before uploading.

`--data-dir` finds the annotation JSON, the training images and the test images
by itself, whatever the JSON is called. Pass `--annotations` and `--image-dir`
instead if your layout is unusual.

Run `inspect_data.py` first and use the `pos_weight` and `patch_size` it reports
for your data: both are set from the class imbalance and the solar radius, and
the defaults will not suit a dataset with different statistics. They can be
passed straight through, e.g. `--patch-size 512 --pos-weight 11.6`.

Swap `--checkpoint …` for `--classical` anywhere above to use the training-free
detector, which needs no weights and runs in a couple of seconds per frame.

Two practical notes:

- The first epoch is slow because every frame is preprocessed; results are
  cached to `--cache-dir` and later epochs read them straight back. It stores
  only what is expensive to recompute, at the smallest precision that does not
  lose information, and splits into photometry per *frame* and targets per
  *annotator record* — so the 447 repeated frames in MAGFiLO are preprocessed
  once, not twice. That is about 2.8 GB for the training split.
- Every pixel-valued setting scales with the measured solar disk — filter
  widths, the merge gap, size limits — so one configuration works whether the
  frames are 512 or 2048 pixels across. These describe properties of the Sun,
  not of the sensor.
- Chirality is read from MAGFiLO's `Left` / `Right` categories, not from a
  field of its own.
- Frames may sit directly in the split directory or nested inside it; both are
  found, matching stems exactly so a record whose image was never distributed
  comes back missing rather than latching on to a similar name.
- Records whose image is not in the split are skipped with a count. Several
  records describing the *same* frame are **kept separate**: in MAGFiLO these
  are independent complete readings by different annotators, which the challenge
  says to treat as different images, so they serve as extra training examples
  that expose the model to genuine annotator disagreement. Train/validation
  splits are grouped by frame, so one image can never appear on both sides.
- Image ids are kept exactly as the dataset gives them. MAGFiLO keys its
  observations by the original GONG frame name (`040301-20140609195854Bh`)
  rather than an integer, and submissions carry those names back unchanged.
- If the distributed JPEGs were resized relative to the frames the annotations
  were drawn on, the loader detects the mismatch, warns once, and **rescales the
  annotations to the image** rather than shrinking the image — barbs do not
  survive downsampling. `inspect_data.py` reports this before you train.

### Tuning the classical detector on your data

Its coverage prior trades recall against precision, and the right value is
dataset-specific. `inspect_data.py` prints a range to search and the exact
command:

```bash
python scripts/tune_classical.py --data-dir data --limit 40
```

---

## How it works

### 1. Photometric preprocessing

The Sun is brighter at disk centre than at the limb, by tens of per cent, and
ground-based observations carry smooth gradients from haze and imperfect flat
fields. Both are removed before anything else happens:

- the limb is located by fitting a circle to the point of steepest intensity
  fall-off along 360 rays, which is accurate to a small fraction of a pixel;
- the radial limb-darkening profile is measured with a **median** in radial bins
  — filaments are a minority of pixels in any bin, so they barely move it — and
  divided out;
- any remaining large-scale gradient is estimated at a scale far coarser than a
  filament and divided out too, which flattens the haze without flattening the
  filaments;
- the image is normalised and inverted, so filaments become bright.

The effect is that filament contrast becomes **uniform across the disk**. On
synthetic data, the filament-to-quiet-Sun contrast measured in three radial
bands is 0.537, 0.531 and 0.549 — near identical, where the raw image varies by
tens of per cent. A single threshold now means the same thing everywhere, and
the network no longer spends capacity learning the radial ramp.

### 2. FilaNet

A U-Net encoder–decoder with residual, group-normalised blocks, plus three
additions aimed at thin structures:

- **Edge-guided bottleneck attention.** A learnable filter bank — initialised to
  Sobel, Laplacian and ridge kernels, so it begins as a genuine edge detector —
  produces an edge map from the input. That map linearly modulates the *Queries*
  and *Keys* of the bottleneck self-attention, so tokens sitting on a boundary
  attend differently from tokens in smooth quiet Sun. Values are left untouched:
  edges should decide what attends to what, not overwrite the content being
  aggregated. The projections are zero-initialised, so the block starts as
  ordinary self-attention and learns how much edge guidance to admit. Because the
  edge map varies with image content, it also removes the need for a learned
  positional encoding.
- **Auxiliary spine and boundary heads.** The network predicts the filament
  centreline and its outline alongside the mask. These are free at inference —
  just ignore them — but they force the shared decoder to represent the skeleton
  and outline explicitly.
- **Deep supervision** on the mask at several decoder scales, giving the encoder
  a short gradient path.

The default model has **9.7M parameters**.

### 3. The loss, which matters more than the architecture

```
total = BCE(weighted) + Tversky + clDice + focal + spine + boundary + deep
```

The important term is **clDice**, computed on differentiable soft skeletons of
the prediction and the target. It scores *topology* rather than area: deleting a
barb breaks a branch of the skeleton, which clDice punishes heavily even though
the pixel count hardly moves. Measured on a synthetic filament whose barb is
13.5% of its pixels, removing that barb raises the clDice loss by **37×** but
the Tversky loss by only 7.7×.

Supporting choices: **Tversky** with `beta > alpha` leans towards recall,
because a missed barb is a hard failure whereas a few over-called edge pixels
are cheap; **per-pixel weights** concentrate cross-entropy on outlines and on
locally thin regions, measured with a distance transform, so a two-pixel barb
gets the full weight bonus and a fat filament body gets almost none; and the
clDice term is **ramped in over the first 500 steps**, since skeletonising a
randomly initialised prediction is meaningless and destabilises early training.

### 4. Inference

Full-resolution overlapping tiles, blended with a raised-cosine window so seams
are invisible, and eight-fold dihedral test-time augmentation. Downsampling the
frame to fit the network is never an option here: it destroys exactly the
structures being scored.

### 5. From pixels to filaments

The challenge asks for each filament as *one coherent object*, which
connected-component labelling does not give you:

- **Fragment merging.** A faint waist or a moment of poor seeing splits one
  filament into two components, costing a hit and adding a false positive. Two
  components are rejoined when an endpoint of each lies within a set distance
  *and* both spines run along the line joining them. Requiring the directions to
  agree is what stops two unrelated filaments that happen to pass close by from
  being welded together.
- **Sunspot rejection by shape**, for detectors that need it. Small, round
  components are removed using second-moment axis ratios, so the test transfers
  across instruments instead of depending on a brightness cutoff. On synthetic
  frames with a heavy sunspot load this cuts the classical detector's false
  discovery rate from 0.60 to 0.14.

  **It is off by default, and that matters.** A trained network has already
  learned to ignore sunspots — it does not predict them at all — so the filter
  finds nothing to remove and instead deletes genuine short, compact filaments.
  Measured on validation frames, switching it on after FilaNet cost 0.145 IoU
  and dropped the hit rate from 0.96 to 0.73. The classical detector enables it
  explicitly because it cannot tell a sunspot from a filament on its own. The
  rule: enable it for a detector that cannot reject sunspots itself, leave it
  off for one that can.

### 6. The classical detector, and its hard limit

No training, no weights. Useful for checking the pipeline runs on your data, and
as a fallback on observations unlike anything in the training set — but **not a
competitive baseline on real observations**, for a reason worth understanding. It scores each pixel by combining a
multi-scale Hessian ridge response — which responds to elongated dark structures
and largely ignores round ones — with the local intensity deficit, then applies
**hysteresis thresholding**: a high threshold seeds confident filament cores,
and those seeds grow outwards through anything above a much lower threshold.
That is what recovers barbs. A barb never exceeds the high threshold alone, but
it is attached to a core that does; an isolated noise excursion of the same
amplitude is not.

Its one real assumption is stated in physical terms rather than hidden in a
percentile: `expected_coverage`, the fraction of the disk covered by filament
cores. This genuinely matters, because filament coverage varies by an order of
magnitude over the solar cycle, and the default is set for real GONG data,
where MAGFiLO's filaments cover about 0.84% of the disk. Synthetic frames are
several times denser and want a higher value. Measure it —
`scripts/inspect_data.py` reports it and `scripts/tune_classical.py` searches
around it — rather than guessing.

> Otsu's method was tried for this and rejected. It assumes two classes of
> comparable size, whereas filaments are a per cent or two of the disk, so it
> over-segments sparse frames badly — on synthetic data a frame with 1.2% true
> coverage was split at 21%.

**Why it cannot win.** On easy synthetic frames, where filaments sit 3.7σ above
the quiet Sun, this reaches IoU 0.62. On realistic frames, where the contrast is
about 1σ and the chromospheric network supplies competing texture, sweeping
*every* threshold on the score map gives a best attainable IoU of **0.083**.
That is a property of the score, not the threshold: at 0.75% prevalence and an
ROC area of 0.75, the top percentile of any such score is dominated by the 99%
background class. `scripts/diagnose_classical.py` measures this ceiling on your
data in about a minute, so you can see it rather than take it on trust. A
network wins here because it pools evidence over a whole neighbourhood and
learns the background texture, neither of which a per-pixel score can do.

---

## Metrics

The challenge ranks on **Panoptic Quality** and the **mean Dice score**, and
additionally assesses fragmentation, over-merging and end-to-end speed.
`filaseg.metrics` implements all of it.

Panoptic Quality is the one that shapes the design:

```
PQ = sum of IoU over matched pairs / (|TP| + 0.5|FP| + 0.5|FN|)
```

It is unforgiving in exactly the way this task needs. Splitting one filament in
two gives a match and a false positive; merging two gives a match and a false
negative. Both cost as much as missing a filament outright, so a prediction can
score well on pixel overlap and badly here — which is why threshold calibration
and checkpoint selection both optimise PQ by default, not IoU. `fragmentation()`
then separates the two failures, since PQ punishes both without saying which
occurred:

| Prediction | PQ | SQ | RQ | Flagged as |
|---|---|---|---|---|
| Exact | 1.000 | 1.000 | 1.000 | — |
| One filament missed | 0.800 | 1.000 | 0.800 | missed |
| One spurious extra | 0.857 | 1.000 | 0.857 | spurious |
| One filament split in two | 0.571 | 1.000 | 0.571 | one-to-many |
| Two filaments merged | 0.400 | 1.000 | 0.400 | many-to-one |
| Every mask shifted 2 px | 0.667 | 0.667 | 1.000 | — |

Also implemented: pixel IoU, precision, recall, `AP@IoU` at several thresholds,
hit and miss rates, pairwise IoU, multi-scale IoU (MSIoU), and clDice as a
direct read-out of whether fine structure survived.

**MSIoU** exists because plain IoU judges thin structures poorly: a filament
three pixels wide predicted one pixel to the left scores near zero despite
being, for any scientific purpose, correct. It compares Sobel edge maps of the
two masks over a ladder of grid resolutions. At fine grids it behaves like
ordinary IoU; at coarse grids a small offset stops mattering, while a structure
that is simply absent still scores nothing.

Measured on a synthetic filament:

| Prediction | IoU | MSIoU | clDice |
|---|---|---|---|
| Exact | 1.000 | 1.000 | 1.000 |
| Shifted one pixel | 0.628 | 0.749 | 0.994 |
| Barb deleted | 0.914 | 0.908 | 0.940 |

Note how the one-pixel shift devastates IoU while MSIoU and clDice correctly
report that the structure is right — and how deleting a barb barely touches IoU,
which is precisely why the loss cannot rely on it.

> The MSIoU implementation follows the published description of the metric
> (Sobel edge maps, grid occupancy at several cell sizes, aggregation across
> scales). If the organisers release reference code with different conventions,
> replace `multiscale_iou` — nothing else depends on its internals.

---

## Results

Full detail in [`docs/results.md`](docs/results.md). Headline comparison, both
detectors on the same held-out synthetic frames:

| Metric | Classical | FilaNet |
|---|---|---|
| IoU | 0.622 | **0.878** |
| clDice | 0.747 | **0.977** |
| MSIoU | 0.625 | **0.863** |
| Hit rate | 0.583 | **0.958** |
| mAP | 0.536 | **0.928** |

**These are synthetic figures, not competition scores.** Real GONG observations
are harder in ways synthetic data cannot capture. They show the pipeline is
correct end to end; re-run the ablation on MAGFiLO before trusting any of the
design conclusions.

Two findings worth carrying over:

- **The clDice term is a trade, not a free gain.** Removing it *raises* pixel
  IoU by 0.023 and *lowers* clDice by 0.011. It spends area agreement to buy
  topological fidelity, which is the right purchase when fine structure is
  scored and the wrong one if you are optimising pixel IoU alone.
- **Edge attention, the auxiliary heads and deep supervision all came out flat**
  on synthetic data — differences of half a per cent or less, inside run-to-run
  noise. That is reported as a null result rather than omitted. Synthetic barbs
  are too clean to test an edge prior properly and the ablation model is small,
  so this benchmark cannot resolve them. Treat them as unproven until the
  ablation is repeated on the real data.

## Repository layout

```
src/filaseg/
  preprocessing/    limb fitting, limb-darkening removal, flattening
  data/             COCO loader, image I/O, datasets, targets, synthetic generator
  models/           FilaNet, edge-guided attention, building blocks
  postprocess/      instance extraction, fragment merging, sunspot rejection
  losses.py         clDice, Tversky, focal, weighted BCE, the combined objective
  metrics.py        IoU, MSIoU, clDice, AP@IoU, hit and miss rates
  classical.py      the training-free detector
  inference.py      tiled full-disk inference with TTA
  train.py          training loop, validation, threshold calibration
  submission.py     COCO, Kaggle RLE and PNG writers

scripts/
  make_synthetic_dataset.py   build a MAGFiLO-shaped synthetic dataset
  train.py                    train FilaNet
  evaluate.py                 score predictions on every challenge metric
  predict.py                  run a detector and write a submission
  tune_classical.py           grid-search the classical detector on real data
  ablation.py                 train variants to show what each component buys
```

## Tests

```bash
python -m pytest                  # everything
python -m pytest -m "not slow"    # skip the ones that train a model
```

182 tests cover geometry, photometry, annotation parsing and encoding, the loss
terms, every metric, instance merging, tiled inference and a full
raw-image-to-metrics run.

## Reproducing the results quoted here

Every number in this README comes from the synthetic generator, which is
seeded and deterministic:

```bash
python scripts/make_synthetic_dataset.py --out data/synthetic --n-images 40 --size 384 --seed 1000
python scripts/evaluate.py --annotations data/synthetic/annotations.json \
    --image-dir data/synthetic/images --classical
python scripts/ablation.py --annotations data/synthetic/annotations.json \
    --image-dir data/synthetic/images --epochs 14 --patch-size 128 --base-width 24 --depth 3
```

**These are synthetic figures and are not competition scores.** They demonstrate
that the pipeline works end to end and that each component earns its place; the
real dataset is harder in ways synthetic data cannot capture, above all the
irregularity of genuine sunspot groups and the variability of ground-based
seeing across the six GONG sites.
