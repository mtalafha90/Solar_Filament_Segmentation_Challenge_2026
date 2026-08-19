# Quick start

Exact commands to get from a fresh machine to a submission file.

## 1. Get the code

```bash
git clone https://github.com/mtalafha90/Solar_Filament_Segmentation_Challenge_2026.git
cd Solar_Filament_Segmentation_Challenge_2026
git checkout claude/solar-filament-segmentation-9bpfy6
```

Already cloned? Pull the branch instead:

```bash
cd Solar_Filament_Segmentation_Challenge_2026
git fetch origin claude/solar-filament-segmentation-9bpfy6
git checkout claude/solar-filament-segmentation-9bpfy6
git pull origin claude/solar-filament-segmentation-9bpfy6
```

## 2. Install

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

Python 3.10 or newer. If you have an NVIDIA GPU and want the CUDA build of
PyTorch, install it from the official index *after* the line above:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

Check it worked:

```bash
python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
```

## 3. Put your data here

Place the dataset at the **root of the repository**, in a folder called `data`:

```
Solar_Filament_Segmentation_Challenge_2026/
├── data/                     <-- create this, or move your existing folder here
│   ├── train/
│   │   ├── xxxxx.jpg
│   │   ├── xxxxx.jpg
│   │   └── MAGFiLO_1.0_Annotations_kaggle2026_train.json
│   └── test/
│       ├── yyyyy.jpg
│       └── yyyyy.jpg
├── src/
├── scripts/
└── README.md
```

`data/` is git-ignored, so nothing large gets committed by accident.

If your folder lives elsewhere, either symlink it:

```bash
ln -s /path/to/your/data data          # macOS / Linux
```

or pass the paths explicitly to every command with `--annotations` and
`--image-dir` instead of relying on the default layout.

The annotation JSON can be called anything — MAGFiLO's is
`MAGFiLO_1.0_Annotations_kaggle2026_train.json`. **You do not need to type it.**
Every script takes `--data-dir data` and finds the JSON, the training images and
the test images itself. If you do pass a name that does not exist, it looks for
the real one next to it and tells you which it used.

## 4. Check the data before training

Always run this first. It verifies that every annotation resolves to a readable
image, that images and annotations agree on size, and it measures the statistics
that set the configuration.

```bash
python scripts/inspect_data.py --data-dir data
```

It finds `train/`, `test/` and the JSON on its own. Read the **SUGGESTED
SETTINGS** block at the end — it prints the `pos_weight` and `patch_size` to
use below, and the coverage range for the classical detector.

## 5. Try the training-free detector first

No training, no GPU, a couple of seconds per frame. This gives you a baseline
score within minutes and confirms the whole pipeline works on your data.

```bash
python scripts/evaluate.py \
    --data-dir data \
    --classical --limit 20
```

## 6. Train FilaNet

```bash
python scripts/train.py --config configs/default.yaml \
    --data-dir data \
    --cache-dir data/cache \
    --output-dir runs/filanet \
    --epochs 60
```

Notes:

- The **first epoch is slow** — every frame is preprocessed and cached to
  `data/cache`. Later epochs read the cache and are much faster.
- Add `--device cpu` if you have no GPU. Expect it to be slow; drop
  `--patch-size` to 128 and `--samples-per-epoch` to a few hundred to test the
  loop before committing to a full run.
- Override anything from `configs/default.yaml` on the command line, e.g.
  `--batch-size 4` if you run out of GPU memory.
- The best checkpoint is written to `runs/filanet/best.pt`, with the decision
  threshold calibrated on validation and stored inside it. Progress is logged to
  `runs/filanet/history.json`.

Resume or retrain by rerunning the same command with a different
`--output-dir`.

## 7. Score it

```bash
python scripts/evaluate.py \
    --data-dir data \
    --cache-dir data/cache \
    --checkpoint runs/filanet/best.pt \
    --out runs/filanet/scores.json
```

Prints IoU, precision, recall, clDice, multi-scale IoU, hit and miss rates and
AP at three IoU thresholds.

## 8. Predict on the test set and write a submission

```bash
python scripts/predict.py \
    --images data/test \
    --checkpoint runs/filanet/best.pt \
    --out submission.json --format coco
```

Use `--format csv` for run-length encoded rows instead, and `--format png` for
label-map images. Swap `--checkpoint runs/filanet/best.pt` for `--classical` to
submit the training-free detector's output.

Check the printed **submission summary** before uploading: if the filaments per
image or the pixel coverage look nothing like the training statistics that
`inspect_data.py` reported, the threshold is wrong.

## 9. Optional: tune the classical detector

Its coverage prior trades recall against precision and is dataset-specific.
`inspect_data.py` prints the range to search:

```bash
python scripts/tune_classical.py \
    --data-dir data \
    --cache-dir data/cache --limit 40
```

## 10. Optional: check the design choices on your data

The ablation in `docs/results.md` was run on synthetic data, and three of the
components came out flat there. Re-run it on the real thing before trusting
those conclusions:

```bash
python scripts/ablation.py \
    --data-dir data \
    --cache-dir data/cache --epochs 40
```

## Troubleshooting

**`no images found under data/test`** — check the extension is one of `.jpg`,
`.jpeg`, `.png`, `.fits`, `.npy`. Anything else is skipped.

**`No annotation JSON found`** — pass `--data-dir data`, or point
`--annotations` straight at the file. If more than one JSON sits in the folder,
the error lists them so you can pick.

**`no image for 'xxx.fits' under data/train`** — the JSON names files that are
not there. The loader already tries the same stem with any known extension, so
this means the names genuinely differ. Check `ls data/train | head`.

**A warning about annotation size not matching image size** — expected if the
distributed JPEGs were resized from the annotated frames. The loader rescales
the annotations to the image automatically and only warns once.

**Out of GPU memory** — lower `--batch-size`, then `--patch-size`.

**Training loss goes to zero and nothing is predicted** — the positive class is
being ignored. Raise `pos_weight` in `configs/default.yaml` to the value
`inspect_data.py` suggested.

**Everything is slow on CPU** — that is expected for training. The classical
detector (`--classical`) needs no training at all and is the right thing to run
first.

## Run the tests

```bash
python -m pytest                  # everything, a couple of minutes
python -m pytest -m "not slow"    # skip the ones that train a model
```
