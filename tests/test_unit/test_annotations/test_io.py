from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from ethology.annotations.io import (
    STANDARD_BBOXES_COLUMNS,
    STANDARD_BBOXES_INDEX,
    _df_bboxes_from_multiple_files,
    _df_bboxes_from_single_file,
    _df_bboxes_from_single_specific_file,
    _df_rows_from_valid_COCO_file,
    _df_rows_from_valid_VIA_file,
    df_bboxes_from_file,
)
from ethology.annotations.validators import ValidCOCO, ValidVIA

# TODO:
# - add fixture for small annotations files
# - add fixture for annotation files with duplicates
# - test kwargs? - should throw error if ignore_index=True is included
# - test multiple supercategories and categories in one file?


@pytest.fixture
def multiple_input_files(annotations_test_data):
    return {
        "VIA": [
            {
                "path": annotations_test_data["VIA_JSON_sample_1.json"],
                "n_annotations": 4440,
                "n_images": 50,
            },
            {
                "path": annotations_test_data["VIA_JSON_sample_2.json"],
                "n_annotations": 3977,
                "n_images": 50,
            },
        ],
        "COCO": [
            {
                "path": annotations_test_data["COCO_JSON_sample_1.json"],
                "n_annotations": 4344,
                "n_images": 100,
            },
            {
                "path": annotations_test_data["COCO_JSON_sample_2.json"],
                "n_annotations": 4618,
                "n_images": 100,
            },
        ],
    }


def assert_dataframe(
    df: pd.DataFrame,
    expected_n_annotations: int,
    expected_n_images: int,
    expected_supercategories: str | list[str],
    expected_categories: str | list[str],
):
    # Check shape of dataframe
    assert df.shape[0] == expected_n_annotations

    # Check annotation_id is the index name and is unique
    assert df.index.name == STANDARD_BBOXES_INDEX
    assert len(set(df.index)) == expected_n_annotations

    # Check number of images
    assert len(df["image_filename"].unique()) == expected_n_images
    assert len(df["image_id"].unique()) == expected_n_images

    # Check minimal columns are present
    assert all([col in df.columns for col in STANDARD_BBOXES_COLUMNS])

    # Check columns are as expected
    assert df.columns.tolist() == [
        "image_filename",
        "image_id",
        "x_min",
        "y_min",
        "width",
        "height",
        "supercategory",
        "category",
    ]

    # check all supercategories are common and equal to expected value
    assert df["supercategory"].unique() == expected_supercategories

    # check all categories are common and equal to expected value
    assert df["category"].unique() == expected_categories

    # check number of annotations per image; df.groupby("image_id").count()


@pytest.mark.parametrize(
    "input_format",
    [
        "VIA",
        "COCO",
    ],
)
@pytest.mark.parametrize(
    "file_path, function_to_mock",
    [
        (
            Path("/path/to/file"),  # single file
            "ethology.annotations.io._df_bboxes_from_single_file",
        ),
        (
            [Path("/path/to/file1"), Path("/path/to/file2")],  # multiple files
            "ethology.annotations.io._df_bboxes_from_multiple_files",
        ),
    ],
)
def test_df_bboxes_from_file_delegation(
    input_format, file_path, function_to_mock
):
    """Test that the general bounding boxes loading function delegates
    correctly.
    """
    # Call general function and see if mocked function is called
    with patch(function_to_mock) as mock:
        df = df_bboxes_from_file(file_path, format=input_format)
        mock.assert_called_once_with(file_path, format=input_format)

    # Check metadata
    assert df.metadata["input_files"] == file_path


@pytest.mark.parametrize(
    "input_format",
    [
        "VIA",
        "COCO",
    ],
)
def test_df_bboxes_from_multiple_files(input_format, multiple_input_files):
    """Test that the general bounding boxes loading function reads
    correctly multiple files.
    """
    # Get format and list of files
    list_files = multiple_input_files[input_format]

    # Get paths, annotations and images
    list_paths = [file["path"] for file in list_files]
    list_n_annotations = [file["n_annotations"] for file in list_files]
    list_n_images = [file["n_images"] for file in list_files]

    # Read all files as a dataframe
    df_all = _df_bboxes_from_multiple_files(list_paths, format=input_format)

    # Check dataframe
    assert_dataframe(
        df_all,
        expected_n_annotations=sum(list_n_annotations),
        expected_n_images=sum(list_n_images),
        expected_supercategories="animal",
        expected_categories="crab",
    )


@pytest.mark.parametrize(
    "input_format, validator, row_function, no_error_expected",
    [
        ("VIA", ValidVIA, _df_rows_from_valid_VIA_file, True),
        ("COCO", ValidCOCO, _df_rows_from_valid_COCO_file, True),
        ("unsupported", None, None, False),
    ],
)
def test_df_bboxes_from_single_file_delegation(
    input_format: str, validator, row_function, no_error_expected: bool
):
    """Test that the _df_bboxes_from_single_file function delegates correctly
    into the specific format readers.
    """
    file_path = Path("/mock/path/to/file")
    function_to_mock = (
        "ethology.annotations.io._df_bboxes_from_single_specific_file"
    )

    # If no error is expected, check that when calling
    # `_df_bboxes_from_single_file`, `_df_bboxes_from_single_specific_file` is
    # called under the hood with the correct arguments
    if no_error_expected:
        with patch(function_to_mock) as mock:
            _df_bboxes_from_single_file(file_path, input_format)
            mock.assert_called_once_with(
                file_path,
                validator=validator,
                df_rows_from_file_fn=row_function,
            )
    else:
        with pytest.raises(ValueError) as excinfo:
            _df_bboxes_from_single_file(file_path, input_format)
        assert "Unsupported format" in str(excinfo.value)


@pytest.mark.parametrize(
    (
        "input_file, validator, row_function, "
        "expected_n_annotations, expected_n_images"
    ),
    [
        (
            "VIA_JSON_sample_1.json",
            ValidVIA,
            _df_rows_from_valid_VIA_file,
            4440,
            50,
        ),
        (
            "VIA_JSON_sample_2.json",
            ValidVIA,
            _df_rows_from_valid_VIA_file,
            3977,
            50,
        ),
        (
            "COCO_JSON_sample_1.json",
            ValidCOCO,
            _df_rows_from_valid_COCO_file,
            4344,
            100,
        ),
        (
            "COCO_JSON_sample_2.json",
            ValidCOCO,
            _df_rows_from_valid_COCO_file,
            4618,
            100,
        ),
    ],
)
def test_df_bboxes_from_single_specific_file(
    input_file: str,
    validator: type[ValidVIA] | type[ValidCOCO],
    row_function: Callable,
    expected_n_annotations: int,
    expected_n_images: int,
    annotations_test_data: dict,
):
    """Test the specific bounding box format readers."""
    # Compute bboxes dataframe from a single file
    df = _df_bboxes_from_single_specific_file(
        file_path=annotations_test_data[input_file],
        validator=validator,
        df_rows_from_file_fn=row_function,
    )

    # Check dataframe
    assert_dataframe(
        df,
        expected_n_annotations,
        expected_n_images,
        expected_supercategories="animal",
        expected_categories="crab",
    )
