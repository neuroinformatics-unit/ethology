"""Tests for ethology/utils/annotation_stats.py."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ethology.utils.annotation_stats import (
    annotations_per_image,
    bbox_size_distribution,
    class_distribution,
    dataset_summary,
)


@pytest.fixture
def simple_ds():
    """Create a small dataset.

    - 2 images
    - image 0: 2 annotations (categories 1 and 3)
    - image 1: 3 annotations (categories 1, 3, 3)
    - image 0 has one padding slot (-1).
    """
    position = np.array(
        [
            [[100.0, 300.0, np.nan], [150.0, 200.0, np.nan]],
            [[50.0, 150.0, 400.0], [80.0, 90.0, 300.0]],
        ]
    )
    shape_arr = np.array(
        [
            [[80.0, 50.0, np.nan], [60.0, 45.0, np.nan]],
            [[40.0, 60.0, 70.0], [35.0, 55.0, 65.0]],
        ]
    )
    image_shape = np.array([[640, 480], [640, 480]])
    category = np.array([[1, 3, -1], [1, 3, 3]])

    return xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                position,
                dims=("image_id", "space", "id"),
                coords={"image_id": [0, 1], "space": ["x", "y"]},
            ),
            "shape": xr.DataArray(
                shape_arr,
                dims=("image_id", "space", "id"),
                coords={"image_id": [0, 1], "space": ["x", "y"]},
            ),
            "image_shape": xr.DataArray(
                image_shape,
                dims=("image_id", "space"),
                coords={"image_id": [0, 1], "space": ["x", "y"]},
            ),
            "category": xr.DataArray(
                category,
                dims=("image_id", "id"),
                coords={"image_id": [0, 1]},
            ),
        },
        attrs={
            "annotation_format": "COCO",
            "map_category_to_str": {1: "Goose", 3: "Mallard"},
            "map_image_id_to_filename": {0: "a.jpg", 1: "b.jpg"},
        },
    )


# ---------------- Class Distribution ----------------


@pytest.mark.parametrize(
    "cat_name, expected_count",
    [
        ("Mallard", 3),
        ("Goose", 2),
    ],
)
def test_category_counts(simple_ds, cat_name, expected_count):
    result = class_distribution(simple_ds)
    assert result[cat_name] == expected_count


def test_class_distribution_returns_series(simple_ds):
    result = class_distribution(simple_ds)
    assert isinstance(result, pd.Series)


def test_class_distribution_total_excludes_padding(simple_ds):
    result = class_distribution(simple_ds)
    assert result.sum() == 5


def test_class_distribution_sorted_most_to_least(simple_ds):
    result = class_distribution(simple_ds)
    assert result.index[0] == "Mallard"


def test_class_distribution_uses_category_names(simple_ds):
    result = class_distribution(simple_ds)
    assert "Mallard" in result.index
    assert "Goose" in result.index


# ---------------- Annotations per image ----------------


@pytest.mark.parametrize(
    "img_id, expected_count",
    [
        (0, 2),
        (1, 3),
    ],
)
def test_annotations_per_image_counts(simple_ds, img_id, expected_count):
    result = annotations_per_image(simple_ds)
    assert result[img_id] == expected_count


def test_annotations_per_image_returns_series(simple_ds):
    result = annotations_per_image(simple_ds)
    assert isinstance(result, pd.Series)


def test_annotations_per_image_length(simple_ds):
    result = annotations_per_image(simple_ds)
    assert len(result) == 2


def test_annotations_per_image_index_is_image_id(simple_ds):
    result = annotations_per_image(simple_ds)
    assert list(result.index) == [0, 1]


# ---------------- Bbox Size Distribution ----------------


def test_bbox_size_returns_dataframe(simple_ds):
    df = bbox_size_distribution(simple_ds)
    assert isinstance(df, pd.DataFrame)


def test_bbox_size_has_required_columns(simple_ds):
    df = bbox_size_distribution(simple_ds)
    assert set(df.columns) == {"width", "height", "area"}


def test_bbox_size_row_count_excludes_padding(simple_ds):
    df = bbox_size_distribution(simple_ds)
    assert len(df) == 5


def test_bbox_size_area_equals_width_times_height(simple_ds):
    df = bbox_size_distribution(simple_ds)
    expected = (df["width"] * df["height"]).reset_index(drop=True)
    actual = df["area"].reset_index(drop=True)
    pd.testing.assert_series_equal(actual, expected, check_names=False)


def test_bbox_size_no_nan_in_output(simple_ds):
    df = bbox_size_distribution(simple_ds)
    assert not df.isnull().any().any()


def test_bbox_size_skips_nan_shapes(simple_ds):
    # NaN width/height slots must be skipped
    ds = simple_ds.copy(deep=True)
    ds["shape"].values[0, 0, 0] = np.nan
    df = bbox_size_distribution(ds)
    assert len(df) == 4


# ---------------- Dataset Summary ----------------


def test_summary_n_images(simple_ds):
    assert dataset_summary(simple_ds)["n_images"] == 2


def test_summary_n_annotations(simple_ds):
    assert dataset_summary(simple_ds)["n_annotations"] == 5


def test_summary_n_categories(simple_ds):
    assert dataset_summary(simple_ds)["n_categories"] == 2


def test_summary_has_required_keys(simple_ds):
    required = {
        "n_images",
        "n_annotations",
        "n_categories",
        "annotations_per_image",
        "class_distribution",
        "bbox_size",
    }
    assert required == set(dataset_summary(simple_ds).keys())


def test_summary_annotations_per_image_has_stats(simple_ds):
    result = dataset_summary(simple_ds)
    for key in ["mean", "std", "min", "max"]:
        assert key in result["annotations_per_image"]


def test_summary_bbox_size_has_stats(simple_ds):
    result = dataset_summary(simple_ds)
    for key in ["mean_width", "mean_height", "mean_area"]:
        assert key in result["bbox_size"]


def test_summary_class_distribution_has_categories(simple_ds):
    result = dataset_summary(simple_ds)
    assert "Goose" in result["class_distribution"]
    assert "Mallard" in result["class_distribution"]


# ---------------- Edge cases ----------------


def test_empty_dataset_summary():
    # Dataset with zero images should return zero counts
    ds = xr.Dataset(
        data_vars={
            "position": xr.DataArray(
                np.empty((0, 2, 0)),
                dims=("image_id", "space", "id"),
                coords={"space": ["x", "y"]},
            ),
            "shape": xr.DataArray(
                np.empty((0, 2, 0)),
                dims=("image_id", "space", "id"),
                coords={"space": ["x", "y"]},
            ),
            "category": xr.DataArray(
                np.empty((0, 0)),
                dims=("image_id", "id"),
            ),
        },
        attrs={"map_category_to_str": {}},
    )
    summary = dataset_summary(ds)
    assert summary["n_images"] == 0
    assert summary["n_annotations"] == 0
    assert summary["bbox_size"]["mean_width"] == 0.0


def test_all_padding_gives_zero_annotations(simple_ds):
    # Dataset where all slots are padding (-1).
    ds = simple_ds.copy(deep=True)
    ds["category"].values[:] = -1
    assert dataset_summary(ds)["n_annotations"] == 0
