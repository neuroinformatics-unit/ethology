"""Tests for the ethology napari reader plugin."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from ethology.napari._reader import (
    _dataset_to_napari_shapes,
    _detect_format,
    _is_coco_file,
    _is_via_file,
    _reader_function,
    napari_get_reader,
)


# ------------- helpers ---------------------------------------------------


@pytest.fixture
def minimal_coco_file(tmp_path: Path) -> Path:
    """Write a minimal valid COCO annotation file to a temp directory."""
    data = {
        "images": [
            {"id": 1, "file_name": "img0.jpg", "width": 200, "height": 100}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "bbox": [10.0, 20.0, 50.0, 30.0],
                "category_id": 1,
                "area": 1500.0,
                "segmentation": [],
                "iscrowd": 0,
            }
        ],
        "categories": [
            {"id": 1, "name": "crab", "supercategory": "animal"}
        ],
    }
    path = tmp_path / "sample.json"
    path.write_text(json.dumps(data))
    return path


@pytest.fixture
def non_annotation_json(tmp_path: Path) -> Path:
    """Write a JSON file that is not an annotation file."""
    path = tmp_path / "other.json"
    path.write_text(json.dumps({"key": "value", "numbers": [1, 2, 3]}))
    return path


# ------------- napari_get_reader -----------------------------------------


def test_get_reader_returns_none_for_non_json_extension(tmp_path: Path):
    txt = tmp_path / "file.txt"
    txt.write_text("hello")
    assert napari_get_reader(str(txt)) is None


def test_get_reader_returns_none_for_non_annotation_json(
    non_annotation_json: Path,
):
    assert napari_get_reader(str(non_annotation_json)) is None


def test_get_reader_returns_callable_for_coco(minimal_coco_file: Path):
    reader = napari_get_reader(str(minimal_coco_file))
    assert callable(reader)


def test_get_reader_accepts_list_input(minimal_coco_file: Path):
    reader = napari_get_reader([str(minimal_coco_file)])
    assert callable(reader)


def test_get_reader_returns_none_for_missing_file(tmp_path: Path):
    missing = str(tmp_path / "does_not_exist.json")
    assert napari_get_reader(missing) is None


# ------------- _is_coco_file / _is_via_file ------------------------------


@pytest.mark.parametrize(
    "data, expected",
    [
        ({"images": [], "annotations": [], "categories": []}, True),
        ({"images": [], "annotations": []}, False),
        ({"_via_img_metadata": {}, "_via_attributes": {}}, False),
        ({}, False),
    ],
)
def test_is_coco_file(data, expected):
    assert _is_coco_file(data) == expected


@pytest.mark.parametrize(
    "data, expected",
    [
        ({"_via_img_metadata": {}, "_via_attributes": {}}, True),
        ({"_via_img_metadata": {}}, False),
        ({"images": [], "annotations": [], "categories": []}, False),
        ({}, False),
    ],
)
def test_is_via_file(data, expected):
    assert _is_via_file(data) == expected


# ------------- _detect_format --------------------------------------------


def test_detect_format_coco(minimal_coco_file: Path):
    assert _detect_format(minimal_coco_file) == "COCO"


def test_detect_format_returns_none_for_non_annotation(
    non_annotation_json: Path,
):
    assert _detect_format(non_annotation_json) is None


def test_detect_format_via(annotations_test_data: dict):
    via_path = annotations_test_data["small_bboxes_VIA.json"]
    assert _detect_format(via_path) == "VIA"


# ------------- _reader_function ------------------------------------------


def test_reader_function_returns_shapes_layer(annotations_test_data: dict):
    result = _reader_function(
        str(annotations_test_data["small_bboxes_COCO.json"])
    )
    assert len(result) == 1
    _shapes, kwargs, layer_type = result[0]
    assert layer_type == "shapes"
    assert "shape_type" in kwargs
    assert "properties" in kwargs
    assert "metadata" in kwargs


def test_reader_function_returns_empty_for_unknown_format(
    non_annotation_json: Path,
):
    assert _reader_function(str(non_annotation_json)) == []


def test_reader_function_accepts_list_input(annotations_test_data: dict):
    path = str(annotations_test_data["small_bboxes_COCO.json"])
    result = _reader_function([path])
    assert len(result) == 1


# ------------- _dataset_to_napari_shapes ---------------------------------


def test_shapes_are_rectangles(annotations_test_data: dict):
    from ethology.io.annotations.load_bboxes import from_files

    ds = from_files(
        annotations_test_data["small_bboxes_COCO.json"], format="COCO"
    )
    shapes, kwargs, layer_type = _dataset_to_napari_shapes(ds)[0]

    assert layer_type == "shapes"
    assert all(st == "rectangle" for st in kwargs["shape_type"])
    for shape in shapes:
        assert shape.shape == (4, 2), "Each rectangle must have 4 corner points"
        unique_xs = np.unique(np.round(shape[:, 1], 8))
        unique_ys = np.unique(np.round(shape[:, 0], 8))
        assert len(unique_xs) == 2, "Rectangle must have exactly 2 unique x values"
        assert len(unique_ys) == 2, "Rectangle must have exactly 2 unique y values"


def test_nan_padded_annotations_are_excluded(annotations_test_data: dict):
    from ethology.io.annotations.load_bboxes import from_files

    ds = from_files(
        annotations_test_data["small_bboxes_COCO.json"], format="COCO"
    )
    n_valid = int((~np.isnan(ds.position.values[:, 0, :])).sum())
    shapes, _kwargs, _layer_type = _dataset_to_napari_shapes(ds)[0]
    assert len(shapes) == n_valid


def test_properties_have_correct_length(annotations_test_data: dict):
    from ethology.io.annotations.load_bboxes import from_files

    ds = from_files(
        annotations_test_data["small_bboxes_COCO.json"], format="COCO"
    )
    shapes, kwargs, _ = _dataset_to_napari_shapes(ds)[0]
    props = kwargs["properties"]
    for key in ("image_id", "annotation_id", "category_id"):
        assert key in props
        assert len(props[key]) == len(shapes)


def test_metadata_contains_required_keys(annotations_test_data: dict):
    from ethology.io.annotations.load_bboxes import from_files

    ds = from_files(
        annotations_test_data["small_bboxes_COCO.json"], format="COCO"
    )
    _shapes, kwargs, _ = _dataset_to_napari_shapes(ds)[0]
    meta = kwargs["metadata"]
    for key in (
        "annotation_format",
        "map_category_to_str",
        "map_image_id_to_filename",
    ):
        assert key in meta


def test_bbox_geometry_is_preserved(annotations_test_data: dict):
    """Test that centre + half-extents round-trip through the reader."""
    from ethology.io.annotations.load_bboxes import from_files

    ds = from_files(
        annotations_test_data["small_bboxes_COCO.json"], format="COCO"
    )
    shapes, _, _ = _dataset_to_napari_shapes(ds)[0]

    x_idx = list(ds.space.values).index("x")
    y_idx = list(ds.space.values).index("y")

    shape_idx = 0
    for img_i in range(ds.sizes["image_id"]):
        for ann_i in range(ds.sizes["id"]):
            x_c = ds.position.values[img_i, x_idx, ann_i]
            if np.isnan(x_c):
                continue
            y_c = ds.position.values[img_i, y_idx, ann_i]
            w = ds.shape.values[img_i, x_idx, ann_i]
            h = ds.shape.values[img_i, y_idx, ann_i]

            rect = shapes[shape_idx]
            rec_x_min = rect[:, 1].min()
            rec_y_min = rect[:, 0].min()
            rec_w = rect[:, 1].max() - rec_x_min
            rec_h = rect[:, 0].max() - rec_y_min

            assert np.isclose(rec_x_min + rec_w / 2, x_c)
            assert np.isclose(rec_y_min + rec_h / 2, y_c)
            assert np.isclose(rec_w, w)
            assert np.isclose(rec_h, h)
            shape_idx += 1