from pathlib import Path
from unittest.mock import patch

import pytest

from ethology.annotations.io import (
    STANDARD_BBOXES_COLUMNS,
    _df_bboxes_from_single_COCO_file,
    _df_bboxes_from_single_file,
    _df_bboxes_from_single_VIA_file,
    df_bboxes_from_file,
)


@pytest.mark.parametrize(
    "input_format, function_to_mock, no_error_expected",
    [
        (
            "VIA",
            "ethology.annotations.io._df_bboxes_from_single_VIA_file",
            True,
        ),
        (
            "COCO",
            "ethology.annotations.io._df_bboxes_from_single_COCO_file",
            True,
        ),
        (
            "unsupported",
            None,
            False,
        ),
    ],
)
def test_df_bboxes_from_single_file(
    input_format, function_to_mock, no_error_expected
):
    """Test that the function delegates to the correct function."""
    file_path = Path("/mock/path/to/file")

    if no_error_expected:
        with patch(function_to_mock) as mock:
            _df_bboxes_from_single_file(file_path, input_format)
            mock.assert_called_once_with(file_path)
    else:
        with pytest.raises(ValueError) as excinfo:
            _df_bboxes_from_single_file(file_path, input_format)
        assert "Unsupported format" in str(excinfo.value)


def test_df_bboxes_from_single_VIA_file(annotations_test_data):
    """Test reading bounding boxes from a single VIA file."""
    # Read a single VIA file
    via_file = annotations_test_data["VIA_JSON_sample_1.json"]
    df = _df_bboxes_from_single_VIA_file(via_file)

    assert df.shape == (4440, 9)
    # Check minimal columns are present
    assert all([col in df.columns for col in STANDARD_BBOXES_COLUMNS])
    # Check exact columns
    assert df.columns.tolist() == [
        "annotation_id",
        "image_filename",
        "image_id",
        "x_min",
        "y_min",
        "width",
        "height",
        "supercategory",
        "category",
    ]

    # check annotation IDs are unique and consecutive?
    # check image_filenames correspond to image IDs in the VIA file?
    # check image_id corresponds to image IDs in the VIA file?
    # check all supercategories are "animal"
    # check all categories are "crab"
    # check number of annotations per image; df.groupby("image_id").count()
    # check number of images
    assert len(df.groupby("image_id")) == 50


def test_df_bboxes_from_single_COCO_file(annotations_test_data):
    """Test reading bounding boxes from a single COCO file."""
    # Read a single COCO file
    # Check: is it supposed to be the same data as the VIA file?
    coco_file = annotations_test_data["COCO_JSON_sample_1.json"]
    df = _df_bboxes_from_single_COCO_file(coco_file)

    assert df.shape == (4344, 9)
    assert all([col in df.columns for col in STANDARD_BBOXES_COLUMNS])
    assert df.columns.tolist() == [
        "annotation_id",
        "image_filename",
        "image_id",
        "x_min",
        "y_min",
        "width",
        "height",
        "supercategory",
        "category",
    ]

    # check annotation IDs are unique and consecutive?
    # check image_filenames correspond to image IDs in the COCO file?
    # check image_id corresponds to image IDs in the COCO file?
    # check all supercategories are "animal"
    # check all categories are "crab"
    # check number of annotations per image; df.groupby("image_id").count()
    # check number of images
    assert len(df.groupby("image_id")) == 100


@pytest.mark.parametrize(
    "input_format",
    [
        "VIA",
        "COCO",
    ],
)
def test_df_bboxes_from_file_single(input_format):
    """Test _df_bboxes_from_file when passing a single file."""
    function_to_mock = "ethology.annotations.io._df_bboxes_from_single_file"
    file_path = Path("/path/to/file")

    with patch(function_to_mock) as mock:
        df_bboxes_from_file(file_path, format=input_format)
        mock.assert_called_once_with(file_path, format=input_format)


# Test for multiple and single files
def test_df_bboxes_from_file_multiple():
    pass
