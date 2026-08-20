"""Tests for string image ids and dataset layout discovery.

Both exist because of real failures on MAGFiLO: its image ids are the original
GONG frame names rather than integers, and its annotation file is called
``MAGFiLO_1.0_Annotations_kaggle2026_train.json``, not ``annotations.json``.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pytest
from conftest import require

from filaseg.data.coco import load_coco, normalise_id, summarise
from filaseg.data.dataset import MagfiloDataset
from filaseg.data.layout import (
    count_images,
    discover,
    find_annotation_files,
    resolve_annotations,
)


GONG_ID = "040301-20140609195854Bh"


@pytest.fixture
def magfilo_style(synthetic_dataset, tmp_path):
    """A dataset with MAGFiLO's real conventions: string ids, odd JSON name."""
    from PIL import Image

    from filaseg.data.io import find_image, read_image

    with (synthetic_dataset / "annotations.json").open() as handle:
        coco = json.load(handle)

    root = tmp_path / "magfilo"
    (root / "train").mkdir(parents=True)
    (root / "test").mkdir(parents=True)

    remap = {}
    for index, entry in enumerate(coco["images"]):
        name = f"04030{index}-2014060919585{index}Bh"
        remap[entry["id"]] = name
        array = read_image(find_image(synthetic_dataset / "images", entry["file_name"]))
        scaled = (255 * np.clip(array / max(array.max(), 1e-6), 0, 1)).astype(np.uint8)
        Image.fromarray(scaled).save(root / "train" / f"{name}.jpeg", quality=95)
        entry["id"] = name
        entry["file_name"] = f"{name}.jpeg"
    for annotation in coco["annotations"]:
        annotation["image_id"] = remap[annotation["image_id"]]
        annotation["id"] = f"ann_{annotation['id']}"

    # One unlabelled test image.
    Image.fromarray(
        np.zeros((32, 32), dtype=np.uint8)
    ).save(root / "test" / "040309-20140609195859Bh.jpeg")

    name = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    with name.open("w") as handle:
        json.dump(coco, handle)
    return root


def test_normalise_id_keeps_strings_and_integers_apart():
    assert normalise_id(GONG_ID) == GONG_ID
    assert normalise_id(5) == 5
    assert normalise_id("7") == 7  # an integer written as text
    assert normalise_id("abc") == "abc"
    assert normalise_id(3.0) == 3
    assert normalise_id("-12") == -12


def test_load_coco_accepts_string_image_ids(magfilo_style):
    annotations = next((magfilo_style / "train").glob("*.json"))
    records, _ = load_coco(annotations)

    assert records
    assert all(isinstance(r.image_id, str) for r in records)
    # Annotations must attach to their image, not be dropped as orphans.
    assert summarise(records)["n_filaments"] > 0
    assert records[0].annotations[0].image_id == records[0].image_id
    assert isinstance(records[0].annotations[0].annotation_id, str)


def test_string_ids_survive_the_dataset_and_its_cache(magfilo_style, tmp_path):
    annotations = next((magfilo_style / "train").glob("*.json"))
    dataset = MagfiloDataset(
        annotations, magfilo_style / "train", cache_dir=tmp_path / "cache"
    )
    cold = dataset[0]
    assert isinstance(cold.image_id, str)
    assert cold.image_id.endswith("Bh")

    warm = dataset[0]  # served from the cache
    assert warm.image_id == cold.image_id
    assert np.array_equal(warm.mask, cold.mask)

    # Cache filenames must be safe: one photometry file per frame, one target
    # file per annotator record.
    frames = list((tmp_path / "cache" / "frames").glob("*.npz"))
    targets = list((tmp_path / "cache" / "targets").glob("*.npz"))
    assert len(frames) == 1 and len(targets) == 1
    assert all("/" not in p.name for p in frames + targets)


def test_cache_paths_do_not_collide_for_similar_ids(tmp_path):
    from filaseg.data.coco import ImageRecord

    dataset = MagfiloDataset.__new__(MagfiloDataset)
    dataset.cache_dir = tmp_path
    a = dataset._cache_path(ImageRecord("a/b", "x.jpg", 1, 1))
    b = dataset._cache_path(ImageRecord("a_b", "x.jpg", 1, 1))
    assert a is not None and b is not None
    assert a.name != b.name  # sanitising must not merge distinct ids


def test_filtering_by_string_id(magfilo_style):
    annotations = next((magfilo_style / "train").glob("*.json"))
    everything = MagfiloDataset(annotations, magfilo_style / "train")
    wanted = everything.image_ids[:1]
    subset = MagfiloDataset(annotations, magfilo_style / "train", image_ids=wanted)
    assert subset.image_ids == wanted


def test_discover_finds_the_whole_layout(magfilo_style):
    layout = discover(magfilo_style)
    assert layout.annotations is not None
    assert layout.annotations.name.startswith("MAGFiLO")
    assert layout.train_dir == magfilo_style / "train"
    assert layout.test_dir == magfilo_style / "test"
    assert len(count_images(layout.train_dir)) > 0
    assert len(count_images(layout.test_dir)) == 1


def test_resolve_annotations_corrects_a_wrong_filename(magfilo_style):
    """The failure that started this: a name copied from an example."""
    wrong = magfilo_style / "train" / "annotations.json"
    with pytest.warns(UserWarning, match="does not exist"):
        resolved = resolve_annotations(wrong, magfilo_style / "train")
    assert resolved.exists()
    assert resolved.name.startswith("MAGFiLO")


def test_resolve_annotations_without_any_hint(magfilo_style):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert resolve_annotations(None, None, magfilo_style).exists()
        assert resolve_annotations(None, magfilo_style / "train").exists()


def test_resolve_annotations_prefers_an_existing_path(magfilo_style):
    real = next((magfilo_style / "train").glob("*.json"))
    assert resolve_annotations(real) == real


def test_resolve_annotations_reports_ambiguity(magfilo_style):
    (magfilo_style / "train" / "other.json").write_text("{}")
    with pytest.raises(FileNotFoundError, match="more than one JSON"):
        resolve_annotations(
            magfilo_style / "train" / "annotations.json", magfilo_style / "train"
        )


def test_resolve_annotations_says_where_it_looked(tmp_path):
    with pytest.raises(FileNotFoundError, match="Looked in"):
        resolve_annotations(tmp_path / "nope.json", tmp_path, tmp_path)


def test_find_annotation_files_orders_by_size(tmp_path):
    (tmp_path / "small.json").write_text("{}")
    (tmp_path / "big.json").write_text(" " * 5000)
    assert find_annotation_files(tmp_path)[0].name == "big.json"


def test_submission_preserves_string_ids(tmp_path):
    from filaseg.submission import write_coco, write_rle_csv

    labels = np.zeros((32, 32), dtype=np.int32)
    labels[4:12, 4:24] = 1

    write_coco(tmp_path / "sub.json", [(GONG_ID, labels, None)])
    entry = json.loads((tmp_path / "sub.json").read_text())[0]
    assert entry["image_id"] == GONG_ID

    write_rle_csv(tmp_path / "sub.csv", [(GONG_ID, labels, None)])
    assert GONG_ID in (tmp_path / "sub.csv").read_text()


# ---------------------------------------------------------------------------
# Image resolution: strictness, split disambiguation, collision detection
# ---------------------------------------------------------------------------


def _touch(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x")


def test_find_image_matches_exactly_and_by_stem(tmp_path):
    from filaseg.data.io import find_image

    _touch(tmp_path / "a.jpeg")
    assert find_image(tmp_path, "a.jpeg").name == "a.jpeg"
    assert find_image(tmp_path, "a.fits").name == "a.jpeg"     # format swapped
    assert find_image(tmp_path, "train/a.jpeg").name == "a.jpeg"  # prefix dropped


def test_find_image_searches_below_but_matches_exactly(tmp_path):
    """Releases nest frames one level down, so subdirectories are searched.

    The safety property is not that we refuse to look, but that we match the
    stem *exactly*: a record whose image was never distributed must come back
    missing rather than latching on to a similarly named frame. Pairing a mask
    with the wrong image is far worse than a missing file.
    """
    from filaseg.data.io import find_image

    _touch(tmp_path / "sub" / "b.jpeg")
    assert find_image(tmp_path, "b.jpeg").name == "b.jpeg"

    # A near-miss must not match, at any depth.
    for name in ("b_extra.jpeg", "bb.jpeg", "b2.jpeg"):
        with pytest.raises(FileNotFoundError):
            find_image(tmp_path, name)

    # And opting out of the search still works.
    with pytest.raises(FileNotFoundError):
        find_image(tmp_path, "b.jpeg", search_subdirectories=False)


def test_find_image_raises_for_a_genuinely_absent_file(tmp_path):
    from filaseg.data.io import find_image

    _touch(tmp_path / "a.jpeg")
    with pytest.raises(FileNotFoundError):
        find_image(tmp_path, "definitely_not_here.jpeg")


def test_resolve_images_reports_missing_and_collisions(tmp_path):
    from filaseg.data.io import resolve_images

    _touch(tmp_path / "x.jpeg")
    resolved, missing, collisions = resolve_images(
        tmp_path, ["x.jpeg", "x.fits", "gone.jpeg"]
    )
    assert missing == ["gone.jpeg"]
    assert len(collisions) == 1                      # x.jpeg and x.fits collide
    assert set(next(iter(collisions.values()))) == {"x.jpeg", "x.fits"}
    assert len(resolved) == 2


def test_resolve_images_disambiguates_split_prefixes(tmp_path):
    """One annotation file covering train and test must not cross-contaminate."""
    from filaseg.data.io import resolve_images

    train = tmp_path / "train"
    _touch(train / "x.jpeg")
    _touch(train / "y.jpeg")

    resolved, missing, collisions = resolve_images(
        train, ["train/x.jpeg", "train/y.jpeg", "test/x.jpeg", "test/y.jpeg"]
    )
    assert not collisions
    assert set(resolved) == {"train/x.jpeg", "train/y.jpeg"}
    assert not missing


def test_dataset_skips_records_with_no_image(tmp_path):
    """MAGFiLO's JSON covers more observations than any one split ships."""
    from PIL import Image

    images = tmp_path / "img"
    images.mkdir()
    for name in ("a", "b"):
        Image.fromarray(
            np.random.default_rng(0).integers(0, 255, (48, 48), dtype=np.uint8)
        ).save(images / f"{name}.jpeg")

    coco = {
        "info": {}, "licenses": [],
        "categories": [{"id": 1, "name": "Left"}],
        "images": [
            {"id": n, "file_name": f"{n}.jpeg", "height": 48, "width": 48}
            for n in ("a", "b", "c")
        ],
        "annotations": [
            {"id": i, "image_id": n, "category_id": 1, "bbox": [1, 1, 6, 6],
             "segmentation": [[1, 1, 8, 1, 8, 8]], "area": 24}
            for i, n in enumerate(("a", "b", "c"), 1)
        ],
    }
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(coco))

    with pytest.warns(UserWarning, match="have no image"):
        dataset = MagfiloDataset(path, images)
    assert len(dataset) == 2
    assert dataset.missing == ["c.jpeg"]


def test_dataset_refuses_to_mispair_on_collision(tmp_path):
    from PIL import Image

    images = tmp_path / "img"
    images.mkdir()
    Image.fromarray(np.zeros((48, 48), dtype=np.uint8)).save(images / "a.jpeg")

    coco = {
        "info": {}, "licenses": [], "categories": [{"id": 1, "name": "Left"}],
        # Two records, different names, both resolving to a.jpeg.
        "images": [
            {"id": "one", "file_name": "a.jpeg", "height": 48, "width": 48},
            {"id": "two", "file_name": "a.fits", "height": 48, "width": 48},
        ],
        "annotations": [],
    }
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(coco))

    with pytest.raises(ValueError, match="more than one annotation record"):
        MagfiloDataset(path, images)


def test_chirality_comes_from_magfilo_categories(tmp_path):
    """MAGFiLO encodes chirality in the category, not a field of its own."""
    coco = {
        "info": {}, "licenses": [],
        "categories": [
            {"supercategory": "filament", "id": 1, "name": "Left"},
            {"supercategory": "filament", "id": 2, "name": "Right"},
            {"supercategory": "filament", "id": 3, "name": "Unidentifiable"},
            {"supercategory": "filament", "id": 4, "name": "Ambiguous"},
        ],
        "images": [{"id": "f1", "file_name": "f1.jpeg", "height": 64, "width": 64}],
        "annotations": [
            {"id": i, "image_id": "f1", "category_id": c, "bbox": [1, 1, 5, 5],
             "segmentation": [[1, 1, 6, 1, 6, 6]], "area": 12}
            for i, c in enumerate((1, 1, 2, 3, 4), 1)
        ],
    }
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(coco))

    records, _ = load_coco(path)
    assert summarise(records)["chirality"] == {
        "unknown": 2, "sinistral": 2, "dextral": 1
    }


def test_explicit_chirality_field_still_wins(tmp_path):
    coco = {
        "info": {}, "licenses": [],
        "categories": [{"id": 1, "name": "Left"}],
        "images": [{"id": "f1", "file_name": "f1.jpeg", "height": 64, "width": 64}],
        "annotations": [
            {"id": 1, "image_id": "f1", "category_id": 1, "chirality": "right",
             "bbox": [1, 1, 5, 5], "segmentation": [[1, 1, 6, 1, 6, 6]], "area": 12}
        ],
    }
    path = tmp_path / "ann.json"
    path.write_text(json.dumps(coco))
    records, _ = load_coco(path)
    assert records[0].annotations[0].chirality == 2


def test_min_area_scales_with_the_solar_disk():
    """40 pixels is a floor on a thumbnail and noise on a 2048-pixel frame."""
    from filaseg.postprocess.instances import InstanceConfig, extract_instances

    yy, xx = np.ogrid[:1024, :1024]
    valid = ((yy - 512) ** 2 + (xx - 512) ** 2) <= 450**2
    probability = np.zeros((1024, 1024), dtype=np.float32)
    probability[500:504, 400:600] = 1.0   # a filament, 800 px
    probability[700:706, 700:708] = 1.0   # noise, 48 px

    scaled = extract_instances(probability, valid, InstanceConfig())
    assert scaled.max() == 1               # noise dropped by the scaled floor

    unscaled = extract_instances(
        probability, valid, InstanceConfig(min_area_fraction=0.0)
    )
    assert unscaled.max() == 2             # 48 px survives a flat floor of 40


# ---------------------------------------------------------------------------
# Nested image directories and repeated frames, both as MAGFiLO ships them
# ---------------------------------------------------------------------------


def _write_frame(path: Path, size: int = 64) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (size, size), dtype=np.uint8)
    ).save(path, quality=95)


def test_resolve_images_finds_frames_nested_one_level_down(tmp_path):
    """Releases often put the frames in an images/ folder inside the split."""
    from filaseg.data.io import resolve_images

    train = tmp_path / "train"
    _write_frame(train / "images" / "20140609195854Bh.jpeg")

    resolved, missing, collisions = resolve_images(
        train, ["20140609195854Bh.jpeg", "20990101000000Bh.jpeg"]
    )
    assert not collisions
    assert missing == ["20990101000000Bh.jpeg"]
    assert resolved["20140609195854Bh.jpeg"].parent.name == "images"


def test_repeated_file_name_resolves_once_without_a_collision(tmp_path):
    from filaseg.data.io import resolve_images

    train = tmp_path / "train"
    _write_frame(train / "images" / "a.jpeg")
    resolved, missing, collisions = resolve_images(train, ["a.jpeg", "a.jpeg"])
    assert not collisions and not missing
    assert len(resolved) == 1


def test_build_image_index_maps_stems_to_files(tmp_path):
    from filaseg.data.io import build_image_index

    _write_frame(tmp_path / "a.jpeg")
    _write_frame(tmp_path / "deep" / "b.jpeg")
    index = build_image_index(tmp_path)
    assert set(index) == {"a", "b"}
    assert index["b"][0].name == "b.jpeg"


def _duplicate_frame_dataset(tmp_path):
    """One frame described by two records, each holding half its filaments."""
    train = tmp_path / "train"
    _write_frame(train / "images" / "frameA.jpeg", size=96)
    coco = {
        "info": {}, "licenses": [],
        "categories": [{"id": 1, "name": "Left"}, {"id": 2, "name": "Right"}],
        "images": [
            {"id": "r1", "file_name": "frameA.jpeg", "height": 96, "width": 96},
            {"id": "r2", "file_name": "frameA.jpeg", "height": 96, "width": 96},
            {"id": "held", "file_name": "not_shipped.jpeg", "height": 96, "width": 96},
        ],
        "annotations": [
            {"id": 1, "image_id": "r1", "category_id": 1, "bbox": [30, 30, 20, 20],
             "segmentation": [[30, 30, 50, 30, 50, 50, 30, 50]], "area": 400},
            {"id": 2, "image_id": "r2", "category_id": 2, "bbox": [55, 30, 20, 20],
             "segmentation": [[55, 30, 75, 30, 75, 50, 55, 50]], "area": 400},
            {"id": 3, "image_id": "held", "category_id": 1, "bbox": [1, 1, 5, 5],
             "segmentation": [[1, 1, 6, 1, 6, 6]], "area": 12},
        ],
    }
    path = train / "MAGFiLO_ann.json"
    path.write_text(json.dumps(coco))
    return path, train


def test_records_describing_one_frame_can_be_merged(tmp_path):
    """Merging remains available for datasets whose records really are partial.

    It is off by default because MAGFiLO's repeated records are independent
    complete readings by different annotators, not parts of one annotation.
    """
    annotations, train = _duplicate_frame_dataset(tmp_path)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        dataset = MagfiloDataset(annotations, train, merge_duplicate_frames=True)

    assert len(dataset) == 1
    assert len(dataset.records[0].annotations) == 2   # both halves kept
    assert dataset.merged == 1
    assert dataset.missing == ["not_shipped.jpeg"]

    messages = " ".join(str(w.message) for w in caught)
    assert "merged" in messages
    assert "have no image" in messages

    # Both filaments must survive into the supervision targets.
    prepared = dataset[0]
    assert prepared.instances.max() == 2


def test_merged_records_keep_their_chirality(tmp_path):
    annotations, train = _duplicate_frame_dataset(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = MagfiloDataset(annotations, train, merge_duplicate_frames=True)
    chirality = sorted(a.chirality for a in dataset.records[0].annotations)
    assert chirality == [1, 2]


def test_dataset_reports_zero_observations_rather_than_guessing(tmp_path):
    """If nothing resolves, say so; do not silently train on an empty set."""
    train = tmp_path / "train"
    train.mkdir()
    coco = {
        "info": {}, "licenses": [], "categories": [{"id": 1, "name": "Left"}],
        "images": [{"id": "x", "file_name": "x.jpeg", "height": 32, "width": 32}],
        "annotations": [],
    }
    path = train / "ann.json"
    path.write_text(json.dumps(coco))
    with pytest.warns(UserWarning, match="have no image"):
        dataset = MagfiloDataset(path, train)
    assert len(dataset) == 0


# ---------------------------------------------------------------------------
# The challenge submission format, and independent annotator records
# ---------------------------------------------------------------------------


def test_challenge_rle_round_trips_losslessly():
    require("pycocotools")
    from filaseg.submission import coco_rle_counts, decode_coco_rle_counts

    rng = np.random.default_rng(0)
    mask = np.zeros((2048, 2048), dtype=bool)
    mask[100:140, 200:600] = True
    mask[120:200, 600:640] = True          # a barb
    mask[rng.integers(0, 2048, 50), rng.integers(0, 2048, 50)] = True

    counts = coco_rle_counts(mask)
    assert np.array_equal(decode_coco_rle_counts(counts), mask)
    # The payload must survive a plain CSV without escaping.
    assert '"' not in counts and "," not in counts and "\n" not in counts


def test_challenge_csv_has_the_required_shape(tmp_path):
    require("pycocotools")
    from filaseg.submission import read_challenge_csv, write_challenge_csv

    labels = np.zeros((2048, 2048), dtype=np.int32)
    labels[100:140, 200:600] = 1
    labels[900:930, 1000:1500] = 2

    rows = write_challenge_csv(
        tmp_path / "submission.csv",
        [
            ("20150125172714Mh.jpeg", labels),
            ("20170501024112Bh.jpeg", np.zeros((2048, 2048), dtype=np.int32)),
        ],
    )
    assert rows == 2

    text = (tmp_path / "submission.csv").read_text().splitlines()
    assert text[0] == "filament_id,segmentation_rle"
    assert text[1].startswith("20150125172714Mh_1,")
    assert text[2].startswith("20150125172714Mh_2,")
    # An image with no detections contributes no rows: the grader matches by
    # overlap, so an empty row would count as a spurious segment.
    assert not any(line.startswith("20170501024112Bh") for line in text)

    recovered = read_challenge_csv(tmp_path / "submission.csv")
    assert set(recovered) == {"20150125172714Mh"}
    assert np.array_equal(recovered["20150125172714Mh"][0], labels == 1)


def test_challenge_csv_refuses_the_wrong_frame_size(tmp_path):
    require("pycocotools")
    from filaseg.submission import write_challenge_csv

    with pytest.raises(ValueError, match="2048"):
        write_challenge_csv(
            tmp_path / "bad.csv", [("x.jpeg", np.zeros((512, 512), dtype=np.int32))]
        )


def test_image_id_strips_the_annotator_prefix():
    from filaseg.submission import image_id_from_name

    assert image_id_from_name("20260901165702Bh.jpeg") == "20260901165702Bh"
    assert image_id_from_name("010401-20160920230134Lh.jpeg") == "20160920230134Lh"
    assert image_id_from_name("some/dir/20150125172714Mh.jpeg") == "20150125172714Mh"
    # A hyphen that is not an annotator batch must be left alone.
    assert image_id_from_name("odd-name.jpeg") == "odd-name"


def _two_annotator_dataset(tmp_path):
    """One frame, annotated independently by two people, as MAGFiLO does."""
    from PIL import Image

    train = tmp_path / "train"
    train.mkdir(parents=True, exist_ok=True)
    Image.fromarray(
        np.random.default_rng(0).integers(0, 255, (96, 96), dtype=np.uint8)
    ).save(train / "20160920230134Lh.jpeg")

    coco = {
        "info": {}, "licenses": [], "categories": [{"id": 1, "name": "Left"}],
        "images": [
            {"id": f"01040{n}-20160920230134Lh",
             "file_name": "20160920230134Lh.jpeg", "height": 96, "width": 96}
            for n in (1, 2)
        ],
        "annotations": [
            {"id": f"ann{n}", "image_id": f"01040{n}-20160920230134Lh",
             "category_id": 1, "bbox": [30, 30, 20, 20],
             "segmentation": [[30, 30, 50, 30, 50, 50, 30, 50]], "area": 400}
            for n in (1, 2)
        ],
    }
    path = train / "ann.json"
    path.write_text(json.dumps(coco))
    return path, train


def test_independent_annotations_are_kept_separate(tmp_path):
    """The organisers say to treat these as different images, so we do."""
    annotations, train = _two_annotator_dataset(tmp_path)

    with pytest.warns(UserWarning, match="different annotator"):
        dataset = MagfiloDataset(annotations, train)

    assert len(dataset) == 2            # not merged into one
    assert dataset.merged == 0
    assert dataset.repeated == 1
    # Each keeps its own annotator's reading, one filament each.
    assert all(len(r.annotations) == 1 for r in dataset.records)


def test_merging_is_available_but_off_by_default(tmp_path):
    annotations, train = _two_annotator_dataset(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        merged = MagfiloDataset(annotations, train, merge_duplicate_frames=True)
    assert len(merged) == 1
    assert len(merged.records[0].annotations) == 2


def test_split_keeps_one_frame_out_of_both_sides(tmp_path):
    """Otherwise the model validates on an image it trained on."""
    require("torch")  # split_ids lives in the training module
    from filaseg.train import split_ids

    groups = ["a.jpg", "a.jpg", "b.jpg", "c.jpg", "c.jpg", "c.jpg", "d.jpg", "e.jpg"]
    train_idx, val_idx = split_ids(len(groups), 0.3, seed=0, groups=groups)

    assert sorted(train_idx + val_idx) == list(range(len(groups)))
    train_groups = {groups[i] for i in train_idx}
    val_groups = {groups[i] for i in val_idx}
    assert not (train_groups & val_groups)
    assert val_groups                      # the split is not degenerate


def test_split_without_groups_is_unchanged():
    require("torch")  # split_ids lives in the training module
    from filaseg.train import split_ids

    train_idx, val_idx = split_ids(20, 0.15, seed=0)
    assert len(train_idx) + len(val_idx) == 20
    assert not set(train_idx) & set(val_idx)


def test_photometry_is_computed_once_per_frame(tmp_path):
    """Several annotators share one image, so its preprocessing is shared too."""
    annotations, train = _two_annotator_dataset(tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = MagfiloDataset(annotations, train, cache_dir=tmp_path / "cache")

    first = dataset[0]
    second = dataset[1]

    # One photometry file for the frame, one target file per annotator record.
    frames = list((tmp_path / "cache" / "frames").glob("*.npz"))
    targets = list((tmp_path / "cache" / "targets").glob("*.npz"))
    assert len(frames) == 1
    assert len(targets) == 2

    # The shared parts must be identical, the annotated parts independent.
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.valid, second.valid)
    assert first.disk.radius == pytest.approx(second.disk.radius)
    assert first.image_id != second.image_id

    # And the targets must survive a reload unchanged.
    reloaded = dataset[0]
    assert np.array_equal(reloaded.instances, first.instances)
    assert np.allclose(reloaded.image, first.image, atol=1e-3)


def test_frame_cache_is_rebuilt_when_missing(tmp_path):
    """Deleting the photometry must not corrupt the annotations that reference it."""
    annotations, train = _two_annotator_dataset(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dataset = MagfiloDataset(annotations, train, cache_dir=tmp_path / "cache")

    original = dataset[0]
    for path in (tmp_path / "cache" / "frames").glob("*.npz"):
        path.unlink()

    rebuilt = dataset[0]
    assert np.allclose(rebuilt.image, original.image, atol=1e-3)
    assert np.array_equal(rebuilt.instances, original.instances)
