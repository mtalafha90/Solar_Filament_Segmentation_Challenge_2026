"""Tests for the COCO loader, mask encoders and dataset plumbing."""

from pathlib import Path

import numpy as np
import pytest

from filaseg.data.coco import (
    decode_segmentation,
    load_coco,
    mask_to_polygons,
    mask_to_rle,
    polygons_to_mask,
    rle_to_mask,
    summarise,
)
from filaseg.data.dataset import MagfiloDataset, prepare_observation
from filaseg.data.io import find_image, read_image
from filaseg.data.targets import boundary_map, distance_weight, spine_heatmap


def test_polygon_rasterisation_area():
    square = [[10.0, 10.0, 30.0, 10.0, 30.0, 30.0, 10.0, 30.0]]
    mask = polygons_to_mask(square, 64, 64)
    assert mask.sum() == pytest.approx(400, rel=0.15)
    assert mask[20, 20]
    assert not mask[5, 5]


def test_polygon_round_trip_preserves_area():
    mask = np.zeros((80, 80), dtype=bool)
    mask[20:60, 30:36] = True  # a filament-like bar
    mask[30:34, 36:52] = True  # a barb
    polygons = mask_to_polygons(mask, tolerance=0.5)
    assert polygons
    back = polygons_to_mask(polygons, 80, 80)
    intersection = np.count_nonzero(back & mask)
    union = np.count_nonzero(back | mask)
    assert intersection / union > 0.9


def test_rle_round_trip_is_exact():
    rng = np.random.default_rng(0)
    mask = rng.random((40, 30)) > 0.7
    encoded = mask_to_rle(mask)
    assert encoded["size"] == [40, 30]
    assert np.array_equal(rle_to_mask(encoded, 40, 30), mask)


def test_rle_round_trip_when_first_pixel_is_set():
    mask = np.ones((8, 8), dtype=bool)
    assert np.array_equal(rle_to_mask(mask_to_rle(mask), 8, 8), mask)


def test_decode_segmentation_accepts_a_flat_polygon():
    flat = [10.0, 10.0, 30.0, 10.0, 30.0, 30.0]
    mask = decode_segmentation(flat, 64, 64)
    assert mask.any()


def test_decode_segmentation_of_none_is_empty():
    assert not decode_segmentation(None, 16, 16).any()


def test_load_coco_reads_everything(synthetic_dataset):
    records, meta = load_coco(synthetic_dataset / "annotations.json")
    assert len(records) == 3
    assert meta["categories"][0]["name"] == "filament"

    stats = summarise(records)
    assert stats["n_filaments"] > 0
    assert stats["annotations_with_spine"] == stats["n_filaments"]
    assert stats["chirality"]["sinistral"] + stats["chirality"]["dextral"] > 0

    record = records[0]
    assert record.instance_map().max() == len(record.annotations)
    assert record.semantic_mask().sum() > 0
    annotation = record.annotations[0]
    assert annotation.spine is not None and annotation.spine.shape[1] == 2
    assert annotation.chirality in (0, 1, 2)


def test_spine_is_parsed_as_row_column():
    from filaseg.data.coco import _parse_spine

    # COCO stores (x, y); the codebase works in (row, column).
    parsed = _parse_spine([[3.0, 7.0], [4.0, 9.0]])
    assert parsed is not None
    assert parsed[0].tolist() == [7.0, 3.0]

    flat = _parse_spine([3.0, 7.0, 4.0, 9.0])
    assert flat is not None and flat[0].tolist() == [7.0, 3.0]


def test_chirality_parsing_accepts_words():
    from filaseg.data.coco import _parse_chirality

    assert _parse_chirality("sinistral") == 1
    assert _parse_chirality("Right") == 2
    assert _parse_chirality(None) == 0
    assert _parse_chirality("nonsense") == 0


def test_find_image_tolerates_a_different_extension(synthetic_dataset):
    path = find_image(synthetic_dataset / "images", "synth_00000.fits")
    assert path.suffix == ".npy"
    assert read_image(path).ndim == 2


def test_find_image_raises_when_missing(synthetic_dataset):
    with pytest.raises(FileNotFoundError):
        find_image(synthetic_dataset / "images", "does_not_exist.fits")


def test_targets_have_the_expected_shape():
    mask = np.zeros((64, 64), dtype=bool)
    mask[28:36, 10:54] = True

    spine = spine_heatmap(mask, sigma=1.5)
    assert spine.max() == pytest.approx(1.0, abs=1e-5)
    # The centreline must run along the middle of the bar.
    assert spine[32, 32] > spine[29, 32]

    boundary = boundary_map(mask, width=2)
    assert boundary[27, 32] > 0  # just outside
    assert boundary[32, 32] == 0  # deep inside

    weights = distance_weight(mask)
    assert weights[mask].mean() > weights[~mask].mean()


def test_spine_heatmap_uses_supplied_points():
    mask = np.zeros((40, 40), dtype=bool)
    mask[18:22, 5:35] = True
    points = np.array([[20.0, 5.0], [20.0, 34.0]], dtype=np.float32)
    heatmap = spine_heatmap(mask, points, sigma=0.0)
    assert heatmap[20, 20] == pytest.approx(1.0)


def test_prepare_observation_builds_consistent_targets(observation):
    prepared = prepare_observation(
        observation.image,
        mask=observation.semantic_mask,
        instances=observation.instance_map,
    )
    assert prepared.image.shape == observation.image.shape
    assert prepared.input_stack().shape == (2,) + observation.image.shape
    # Annotations must never extend beyond the solar disk.
    assert not (prepared.mask & ~prepared.valid).any()
    assert prepared.instances.max() > 0
    assert prepared.mu[~prepared.valid].max() == pytest.approx(0.0)


def test_dataset_cache_returns_identical_results(synthetic_dataset, tmp_path):
    dataset = MagfiloDataset(
        synthetic_dataset / "annotations.json",
        synthetic_dataset / "images",
        cache_dir=tmp_path / "cache",
    )
    first = dataset[0]
    second = dataset[0]  # served from the cache this time
    assert np.array_equal(first.image, second.image)
    assert np.array_equal(first.mask, second.mask)
    assert first.disk.radius == pytest.approx(second.disk.radius)
    assert first.file_name == second.file_name


def test_patch_dataset_yields_usable_batches(synthetic_dataset):
    torch = pytest.importorskip("torch")
    from filaseg.data.dataset import FilamentPatchDataset

    source = MagfiloDataset(
        synthetic_dataset / "annotations.json", synthetic_dataset / "images"
    )
    patches = FilamentPatchDataset(source, patch_size=96, samples_per_epoch=8, seed=0)
    assert len(patches) == 8

    sample = patches[0]
    assert sample["input"].shape == (2, 96, 96)
    assert sample["mask"].shape == (1, 96, 96)
    for key, tensor in sample.items():
        assert torch.isfinite(tensor).all(), key

    # Most sampled patches should contain filament pixels.
    positives = sum(1 for i in range(8) if patches[i]["mask"].sum() > 0)
    assert positives >= 6


def test_rescale_record_scales_every_field():
    from filaseg.data.coco import FilamentAnnotation, ImageRecord, rescale_record

    record = ImageRecord(
        image_id=1,
        file_name="a.jpg",
        height=100,
        width=100,
        annotations=[
            FilamentAnnotation(
                annotation_id=1,
                image_id=1,
                bbox=(10.0, 20.0, 30.0, 40.0),
                segmentation=[[10, 20, 40, 20, 40, 60, 10, 60]],
                area=1200.0,
                spine=np.array([[20.0, 10.0], [60.0, 40.0]], dtype=np.float32),
            )
        ],
    )
    before = record.semantic_mask().sum()
    rescale_record(record, 200, 200)

    assert record.height == 200 and record.width == 200
    assert record.annotations[0].bbox == (20.0, 40.0, 60.0, 80.0)
    assert record.annotations[0].area == pytest.approx(4800.0)
    assert record.annotations[0].spine[0].tolist() == [40.0, 20.0]
    # Rasterised area should grow with the square of the scale factor.
    assert record.semantic_mask().sum() == pytest.approx(4 * before, rel=0.05)


def test_rescale_record_is_a_no_op_at_the_same_size():
    from filaseg.data.coco import FilamentAnnotation, ImageRecord, rescale_record

    record = ImageRecord(1, "a.jpg", 64, 64, [
        FilamentAnnotation(1, 1, (1.0, 2.0, 3.0, 4.0), [[1, 2, 5, 2, 5, 8]], area=9.0)
    ])
    rescale_record(record, 64, 64)
    assert record.annotations[0].bbox == (1.0, 2.0, 3.0, 4.0)


def test_rescale_record_handles_non_square_scaling():
    from filaseg.data.coco import FilamentAnnotation, ImageRecord, rescale_record

    record = ImageRecord(1, "a.jpg", 100, 50, [
        FilamentAnnotation(1, 1, (10.0, 10.0, 10.0, 10.0), [[10, 10, 20, 10, 20, 20]])
    ])
    rescale_record(record, 200, 200)
    # Columns scale by 4, rows by 2.
    assert record.annotations[0].bbox == (40.0, 20.0, 40.0, 20.0)


def test_dataset_rescales_when_image_and_annotation_sizes_differ(
    synthetic_dataset, tmp_path
):
    """A JPEG release resized from the annotated frames must still line up."""
    import json

    from PIL import Image

    from filaseg.data.io import find_image, read_image

    with (synthetic_dataset / "annotations.json").open() as handle:
        coco = json.load(handle)

    # Write half-size JPEGs and leave the annotations at the original size.
    out = tmp_path / "small"
    (out / "images").mkdir(parents=True)
    for entry in coco["images"]:
        array = read_image(find_image(synthetic_dataset / "images", entry["file_name"]))
        scaled = (255 * np.clip(array / max(array.max(), 1e-6), 0, 1)).astype(np.uint8)
        half = Image.fromarray(scaled).resize(
            (array.shape[1] // 2, array.shape[0] // 2), Image.BILINEAR
        )
        half.save(out / "images" / f"{Path(entry['file_name']).stem}.jpg", quality=95)
    with (out / "annotations.json").open("w") as handle:
        json.dump(coco, handle)

    with pytest.warns(UserWarning, match="does not match"):
        dataset = MagfiloDataset(out / "annotations.json", out / "images")
        prepared = dataset[0]

    # The mask must land on the half-size grid and still cover a sane area.
    assert prepared.mask.shape == prepared.image.shape
    assert prepared.mask.shape[0] == coco["images"][0]["height"] // 2
    assert prepared.mask.sum() > 0
    assert not (prepared.mask & ~prepared.valid).any()
