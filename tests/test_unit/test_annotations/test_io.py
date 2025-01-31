from collections.abc import Callable
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from ethology.annotations.io import (
    STANDARD_BBOXES_DF_COLUMNS,
    STANDARD_BBOXES_DF_INDEX,
    _df_bboxes_from_multiple_files,
    _df_bboxes_from_single_file,
    _df_bboxes_from_single_specific_file,
    _df_rows_from_valid_COCO_file,
    _df_rows_from_valid_VIA_file,
    df_bboxes_from_files,
)
from ethology.annotations.validators import ValidCOCO, ValidVIA


@pytest.fixture
def multiple_input_files(annotations_test_data: dict) -> dict:
    """Fixture that returns for each format, a pair of annotation files
    with their number of annotations and images.
    """
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
    expected_annots_per_image: int | None = None,
):
    """Check that the dataframe has the expected shape and content."""
    # Check shape of dataframe
    assert df.shape[0] == expected_n_annotations

    # Check annotation_id is the index name, and that IDs are unique
    assert df.index.name == STANDARD_BBOXES_DF_INDEX
    assert len(set(df.index)) == expected_n_annotations

    # Check number of images
    assert len(df["image_filename"].unique()) == expected_n_images
    assert len(df["image_id"].unique()) == expected_n_images

    # Check columns are as expected
    assert df.columns.tolist() == STANDARD_BBOXES_DF_COLUMNS

    # Check supercategories are as expected
    assert df["supercategory"].unique() == expected_supercategories

    # Check categories are as expected
    assert df["category"].unique() == expected_categories

    # Check number of annotations per image if provided
    if expected_annots_per_image:
        assert all(
            df.groupby("image_id").count()["x_min"]
            == expected_annots_per_image
        )  # count number of "x_min" values when grouping by "image_id"


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
def test_df_bboxes_from_file(input_format, file_path, function_to_mock):
    """Test that the general bounding boxes loading function delegates
    correctly to the single or multiple file readers.
    """
    # Call general function and see if mocked function is called
    with patch(function_to_mock) as mock:
        df = df_bboxes_from_files(file_path, format=input_format)
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
    correctly multiple files of the supported formats.
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
def test_df_bboxes_from_single_file(
    input_format: str, validator, row_function, no_error_expected: bool
):
    """Test that the ``_df_bboxes_from_single_file`` function delegates
    correctly into the specific format readers.
    """
    file_path = Path("/mock/path/to/file")
    function_to_mock = (
        "ethology.annotations.io._df_bboxes_from_single_specific_file"
    )

    # If the format is supported, check that when calling
    # `_df_bboxes_from_single_file`, `_df_bboxes_from_single_specific_file` is
    # called under the hood with the correct arguments
    if no_error_expected:
        with patch(function_to_mock) as mock:
            _df_bboxes_from_single_file(file_path, input_format)
            mock.assert_called_once_with(
                file_path,
                validator=validator,
                get_rows_from_file=row_function,
            )
    # If the format is not supported, check that an error is raised
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
        ),  # medium VIA file
        (
            "VIA_JSON_sample_2.json",
            ValidVIA,
            _df_rows_from_valid_VIA_file,
            3977,
            50,
        ),  # medium VIA file
        (
            "small_bboxes_VIA.json",
            ValidVIA,
            _df_rows_from_valid_VIA_file,
            3,
            3,
        ),  # small VIA file
        (
            "COCO_JSON_sample_1.json",
            ValidCOCO,
            _df_rows_from_valid_COCO_file,
            4344,
            100,
        ),  # medium COCO file
        (
            "COCO_JSON_sample_2.json",
            ValidCOCO,
            _df_rows_from_valid_COCO_file,
            4618,
            100,
        ),  # medium COCO file
        (
            "small_bboxes_COCO.json",
            ValidCOCO,
            _df_rows_from_valid_COCO_file,
            3,
            3,
        ),  # small COCO file
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
        get_rows_from_file=row_function,
    )

    # Check dataframe
    # (we only check annotations per image in small datasets)
    assert_dataframe(
        df,
        expected_n_annotations,
        expected_n_images,
        expected_supercategories="animal",
        expected_categories="crab",
        expected_annots_per_image=1 if expected_n_images < 5 else None,
    )


@pytest.mark.parametrize(
    "input_file, expected_n_annotations",
    [
        ("VIA_JSON_sample_1.json", 4440),
        ("VIA_JSON_sample_2.json", 3977),
        ("small_bboxes_VIA.json", 3),
        ("small_bboxes_duplicates_VIA.json", 4),  # contains duplicates
    ],
)
def test_df_rows_from_valid_VIA_file(
    input_file: str,
    expected_n_annotations: int,
    annotations_test_data: dict,
):
    """Test the extraction of rows from a valid VIA file."""
    rows = _df_rows_from_valid_VIA_file(
        file_path=annotations_test_data[input_file]
    )

    # Check number of rows
    assert len(rows) == expected_n_annotations

    # Check each row contains required column data
    for row in rows:
        assert all(
            key in row
            for key in [STANDARD_BBOXES_DF_INDEX] + STANDARD_BBOXES_DF_COLUMNS
        )


@pytest.mark.parametrize(
    "input_file, expected_n_annotations",
    [
        ("COCO_JSON_sample_1.json", 4344),
        ("COCO_JSON_sample_2.json", 4618),
        ("small_bboxes_COCO.json", 3),
        ("small_bboxes_duplicates_COCO.json", 4),  # contains duplicates
    ],
)
def test_df_rows_from_valid_COCO_file(
    input_file: str,
    expected_n_annotations: int,
    annotations_test_data: dict,
):
    """Test the extraction of rows from a valid COCO file."""
    rows = _df_rows_from_valid_COCO_file(
        file_path=annotations_test_data[input_file]
    )

    # Check number of rows
    assert len(rows) == expected_n_annotations

    # Check each row contains required column data
    for row in rows:
        assert all(
            key in row
            for key in [STANDARD_BBOXES_DF_INDEX] + STANDARD_BBOXES_DF_COLUMNS
        )


@pytest.mark.parametrize(
    ("input_file, validator, row_function"),
    [
        (
            "small_bboxes_duplicates_VIA.json",
            ValidVIA,
            _df_rows_from_valid_VIA_file,
        ),
        (
            "small_bboxes_duplicates_COCO.json",
            ValidCOCO,
            _df_rows_from_valid_COCO_file,
        ),
    ],
)
def test_df_bboxes_from_single_specific_file_duplicates(
    input_file: str,
    validator: type[ValidVIA] | type[ValidCOCO],
    row_function: Callable,
    annotations_test_data: dict,
):
    """Test the specific bounding box format readers when the input file
    contains duplicate annotations.
    """
    # Properties of input data
    # one annotation is duplicated in the first frame
    expected_n_annotations_w_duplicates = 4
    expected_n_annotations_wo_duplicates = 3
    expected_n_images = 3

    # Extract rows
    rows = row_function(file_path=annotations_test_data[input_file])

    # Check total number of annotations including duplicates
    assert len(rows) == expected_n_annotations_w_duplicates

    # Compute bboxes dataframe
    df = _df_bboxes_from_single_specific_file(
        file_path=annotations_test_data[input_file],
        validator=validator,
        get_rows_from_file=row_function,
    )

    # Check dataframe has no duplicates
    assert_dataframe(
        df,
        expected_n_annotations_wo_duplicates,
        expected_n_images,
        expected_supercategories="animal",
        expected_categories="crab",
    )
