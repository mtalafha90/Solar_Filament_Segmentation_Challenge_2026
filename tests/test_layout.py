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

    # Cache filenames must be safe, and one per observation.
    written = list((tmp_path / "cache").glob("*.npz"))
    assert len(written) == 1
    assert "/" not in written[0].name


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
