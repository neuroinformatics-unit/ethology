"""Tests for the ethology napari writer plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from ethology.napari._writer import _shapes_to_dataset, write_shapes

# ------------- helpers ---------------------------------------------------


def _make_rect(x_min: float, y_min: float, x_max: float, y_max: float):
    """Return a napari-style rectangle array (4 corners in y-x order)."""
    return np.array(
        [
            [y_min, x_min],
            [y_min, x_max],
            [y_max, x_max],
            [y_max, x_min],
        ],
        dtype=np.float64,
    )


@pytest.fixture
def two_image_shapes():
    """Two rectangles, one per image."""
    return [
        _make_rect(10, 20, 70, 60),  # image 0: x in [20,60], y in [10,70]
        _make_rect(100, 150, 250, 350),  # image 1
    ]


@pytest.fixture
def two_image_meta(two_image_shapes):
    """Metadata compatible with two_image_shapes."""
    return {
        "shape_type": ["rectangle", "rectangle"],
        "properties": {
            "image_id": np.array([0, 1], dtype=int),
            "annotation_id": np.array([0, 0], dtype=int),
            "category_id": np.array([1, 1], dtype=int),
        },
        "metadata": {
            "annotation_format": "COCO",
            "annotation_files": "",
            "images_directories": None,
            "map_category_to_str": {1: "crab"},
            "map_image_id_to_filename": {
                0: "img0.jpg",
                1: "img1.jpg",
            },
        },
    }


# ------------- _shapes_to_dataset ----------------------------------------


def test_shapes_to_dataset_returns_valid_dataset(
    two_image_shapes, two_image_meta
):
    ds = _shapes_to_dataset(
        two_image_shapes,
        two_image_meta["shape_type"],
        two_image_meta["properties"],
        two_image_meta["metadata"],
    )
    assert isinstance(ds, xr.Dataset)
    for var in ("position", "shape", "category"):
        assert var in ds.data_vars
    assert set(ds.dims) >= {"image_id", "space", "id"}


def test_shapes_to_dataset_correct_centre_and_size(
    two_image_shapes, two_image_meta
):
    """Verify that the writer correctly recovers bbox centres from corners."""
    ds = _shapes_to_dataset(
        two_image_shapes,
        two_image_meta["shape_type"],
        two_image_meta["properties"],
        two_image_meta["metadata"],
    )
    # First rect: _make_rect(x_min=10, y_min=20, x_max=70, y_max=60)
    # → x centre = (10+70)/2 = 40, width  = 70-10 = 60
    # → y centre = (20+60)/2 = 40, height = 60-20 = 40
    x_c = ds.position.sel(space="x").values[0, 0]
    y_c = ds.position.sel(space="y").values[0, 0]
    w = ds.shape.sel(space="x").values[0, 0]
    h = ds.shape.sel(space="y").values[0, 0]

    assert np.isclose(x_c, 40.0)
    assert np.isclose(y_c, 40.0)
    assert np.isclose(w, 60.0)
    assert np.isclose(h, 40.0)


def test_shapes_to_dataset_non_rectangles_are_skipped(two_image_meta):
    shapes = [
        _make_rect(10, 20, 60, 70),
        np.array([[0, 0], [10, 5], [20, 0]], dtype=float),  # triangle
    ]
    shape_types = ["rectangle", "polygon"]
    ds = _shapes_to_dataset(
        shapes,
        shape_types,
        two_image_meta["properties"],
        two_image_meta["metadata"],
    )
    # Only the rectangle (image_id=0) is kept
    assert ds.sizes["image_id"] == 1


def test_shapes_to_dataset_raises_if_no_rectangles():
    with pytest.raises(ValueError, match="No rectangle shapes"):
        _shapes_to_dataset(
            [np.array([[0, 0], [10, 5], [20, 0]], dtype=float)],
            ["polygon"],
            {"image_id": np.array([0])},
            {},
        )


def test_shapes_to_dataset_builds_fallback_maps():
    """When no metadata maps are provided, fallback names are generated."""
    shapes = [_make_rect(0, 0, 10, 10)]
    ds = _shapes_to_dataset(
        shapes,
        ["rectangle"],
        {"image_id": np.array([0]), "category_id": np.array([2])},
        {},
    )
    assert 0 in ds.attrs["map_image_id_to_filename"]
    assert 2 in ds.attrs["map_category_to_str"]


def test_shapes_to_dataset_category_ids_stored(
    two_image_shapes, two_image_meta
):
    ds = _shapes_to_dataset(
        two_image_shapes,
        two_image_meta["shape_type"],
        two_image_meta["properties"],
        two_image_meta["metadata"],
    )
    assert ds.category.values[0, 0] == 1
    assert ds.category.values[1, 0] == 1


# ------------- write_shapes ----------------------------------------------


def test_write_shapes_creates_coco_json(
    two_image_shapes, two_image_meta, tmp_path: Path
):
    out = str(tmp_path / "out.json")
    result = write_shapes(out, two_image_shapes, two_image_meta)

    assert result == [out]
    assert Path(out).exists()

    with open(out) as f:
        data = json.load(f)

    for section in ("images", "annotations", "categories"):
        assert section in data
    assert len(data["annotations"]) == 2


def test_write_shapes_annotation_count(
    two_image_shapes, two_image_meta, tmp_path: Path
):
    out = str(tmp_path / "out.json")
    write_shapes(out, two_image_shapes, two_image_meta)
    with open(out) as f:
        data = json.load(f)
    assert len(data["annotations"]) == len(two_image_shapes)


# ------------- round-trip (reader → writer → re-load) --------------------


def test_roundtrip_annotation_count(
    annotations_test_data: dict, tmp_path: Path
):
    """Loading then saving must preserve the number of bounding boxes."""
    from ethology.io.annotations.load_bboxes import from_files
    from ethology.napari._reader import _dataset_to_napari_shapes

    input_file = annotations_test_data["small_bboxes_COCO.json"]
    ds = from_files(input_file, format="COCO")
    shapes, kwargs, _ = _dataset_to_napari_shapes(ds)[0]

    meta = {
        "shape_type": kwargs["shape_type"],
        "properties": kwargs["properties"],
        "metadata": kwargs["metadata"],
    }

    out = str(tmp_path / "roundtrip.json")
    write_shapes(out, shapes, meta)

    ds2 = from_files(out, format="COCO")
    n_orig = int((~np.isnan(ds.position.values[:, 0, :])).sum())
    n_reloaded = int((~np.isnan(ds2.position.values[:, 0, :])).sum())
    assert n_orig == n_reloaded


def test_roundtrip_bbox_geometry(annotations_test_data: dict, tmp_path: Path):
    """Bounding box coordinates must be identical after a round-trip."""
    from ethology.io.annotations.load_bboxes import from_files
    from ethology.napari._reader import _dataset_to_napari_shapes

    input_file = annotations_test_data["small_bboxes_COCO.json"]
    ds = from_files(input_file, format="COCO")
    shapes, kwargs, _ = _dataset_to_napari_shapes(ds)[0]
    meta = {
        "shape_type": kwargs["shape_type"],
        "properties": kwargs["properties"],
        "metadata": kwargs["metadata"],
    }

    out = str(tmp_path / "roundtrip_geom.json")
    write_shapes(out, shapes, meta)
    ds2 = from_files(out, format="COCO")

    # Compare non-NaN position values (order may differ per image)
    for img_i in range(ds.sizes["image_id"]):
        orig_pos = ds.position.values[img_i]  # (2, n_annots)
        mask = ~np.isnan(orig_pos[0])
        orig_sorted = np.sort(orig_pos[:, mask], axis=1)

        img_id = int(ds.image_id.values[img_i])
        img_i2 = list(ds2.image_id.values).index(img_id)
        reloaded_pos = ds2.position.values[img_i2]
        mask2 = ~np.isnan(reloaded_pos[0])
        reloaded_sorted = np.sort(reloaded_pos[:, mask2], axis=1)

        np.testing.assert_allclose(orig_sorted, reloaded_sorted, atol=1e-6)
