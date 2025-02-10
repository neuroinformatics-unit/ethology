from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pandas as pd
import pytest

import ethology
from ethology.annotations.io.load_bboxes import (
    STANDARD_BBOXES_DF_COLUMNS,
    STANDARD_BBOXES_DF_INDEX,
    _df_rows_from_valid_COCO_file,
    _df_rows_from_valid_VIA_file,
    _from_multiple_files,
    _from_single_file,
    from_files,
)


@pytest.fixture
def multiple_files(annotations_test_data: dict) -> dict:
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


@pytest.fixture
def multiple_files_duplicates(annotations_test_data: dict) -> dict:
    """Fixture that returns for each format, a pair of annotation files
    with duplicate annotations across the two files.
    """
    return {
        "VIA": {
            "files": [
                {
                    "path": annotations_test_data["small_bboxes_VIA.json"],
                    "n_annotations": 3,
                },
                {
                    "path": annotations_test_data[
                        "small_bboxes_VIA_subset.json"
                    ],
                    "n_annotations": 2,
                },
            ],
            "duplicates": 1,
            "n_images": 3,
        },
        "COCO": {
            "files": [
                {
                    "path": annotations_test_data["small_bboxes_COCO.json"],
                    "n_annotations": 3,
                },
                {
                    "path": annotations_test_data[
                        "small_bboxes_COCO_subset.json"
                    ],
                    "n_annotations": 2,
                },
            ],
            "duplicates": 1,
            "n_images": 3,
        },
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
    "images_dirs",
    [
        [Path("/path/to/images")],  # single directory
        [Path("/path/to/images1"), Path("/path/to/images2")],  # multiple dirs
        None,  # no images directories
    ],
)
@pytest.mark.parametrize(
    "file_path, function_to_mock",
    [
        (
            Path("/path/to/file"),  # single file
            "ethology.annotations.io.load_bboxes._from_single_file",
        ),
        (
            [Path("/path/to/file1"), Path("/path/to/file2")],  # multiple files
            "ethology.annotations.io.load_bboxes._from_multiple_files",
        ),
    ],
)
def test_from_files(
    input_format: Literal["VIA", "COCO"],
    images_dirs: Path | str | list[Path | str] | None,
    file_path: Path,
    function_to_mock: str,
):
    """Test that the general bounding boxes loading function delegates
    correctly to the single or multiple file readers, and check the
    metadata is added correctly.
    """
    # Call general function and see if mocked function is called
    with patch(function_to_mock) as mock:
        df = from_files(
            file_path,
            format=input_format,
            images_dirs=images_dirs,
        )
        mock.assert_called_once_with(file_path, format=input_format)

    # Check metadata
    assert df.attrs["input_files"] == file_path
    assert df.attrs["format"] == input_format
    if images_dirs:
        assert df.attrs["images_dirs"] == images_dirs


@pytest.mark.parametrize(
    "input_format",
    [
        "VIA",
        "COCO",
    ],
)
def test_from_multiple_files(
    input_format: Literal["VIA", "COCO"], multiple_files: dict
):
    """Test that the general bounding boxes loading function reads
    correctly multiple files of the supported formats.
    """
    # Get format and list of files
    list_files = multiple_files[input_format]

    # Get paths, annotations and images
    list_paths = [file["path"] for file in list_files]
    list_n_annotations = [file["n_annotations"] for file in list_files]
    list_n_images = [file["n_images"] for file in list_files]

    # Read all files as a dataframe
    df_all = _from_multiple_files(list_paths, format=input_format)

    # Check dataframe
    assert_dataframe(
        df_all,
        expected_n_annotations=sum(list_n_annotations),
        expected_n_images=sum(list_n_images),
        expected_supercategories="animal",
        expected_categories="crab",
    )


def test_from_single_file_unsupported():
    """Test that unsupported formats throw the expected errors."""
    file_path = Path("/mock/path/to/file")
    input_format = "unsupported"

    with pytest.raises(ValueError) as excinfo:
        _from_single_file(file_path=file_path, format=input_format)
    assert "Unsupported format" in str(excinfo.value)


@pytest.mark.parametrize(
    ("input_file, format, expected_n_annotations, expected_n_images"),
    [
        (
            "VIA_JSON_sample_1.json",
            "VIA",
            4440,
            50,
        ),  # medium VIA file
        (
            "VIA_JSON_sample_2.json",
            "VIA",
            3977,
            50,
        ),  # medium VIA file
        (
            "small_bboxes_VIA.json",
            "VIA",
            3,
            3,
        ),  # small VIA file
        (
            "COCO_JSON_sample_1.json",
            "COCO",
            4344,
            100,
        ),  # medium COCO file
        (
            "COCO_JSON_sample_2.json",
            "COCO",
            4618,
            100,
        ),  # medium COCO file
        (
            "small_bboxes_COCO.json",
            "COCO",
            3,
            3,
        ),  # small COCO file
    ],
)
def test_from_single_file(
    input_file: str,
    format: Literal["VIA", "COCO"],
    expected_n_annotations: int,
    expected_n_images: int,
    annotations_test_data: dict,
):
    """Test the specific bounding box format readers."""
    # Compute bboxes dataframe from a single file
    df = _from_single_file(
        file_path=annotations_test_data[input_file],
        format=format,
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
    ("input_file, format"),
    [
        ("small_bboxes_duplicates_VIA.json", "VIA"),
        ("small_bboxes_duplicates_COCO.json", "COCO"),
    ],
)
def test_from_single_file_duplicates(
    input_file: str,
    format: Literal["VIA", "COCO"],
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
    row_function = getattr(
        ethology.annotations.io.load_bboxes,
        f"_df_rows_from_valid_{format}_file",
    )
    rows = row_function(file_path=annotations_test_data[input_file])

    # Check total number of annotations including duplicates
    assert len(rows) == expected_n_annotations_w_duplicates

    # Compute bboxes dataframe
    df = _from_single_file(
        file_path=annotations_test_data[input_file],
        format=format,
    )

    # Check dataframe has no duplicates
    assert_dataframe(
        df,
        expected_n_annotations_wo_duplicates,
        expected_n_images,
        expected_supercategories="animal",
        expected_categories="crab",
    )


@pytest.mark.parametrize(
    ("input_file, format, expected_exception"),
    [
        (
            "small_bboxes_no_cat_VIA.json",
            "VIA",
            does_not_raise(),
        ),
        (
            "small_bboxes_no_cat_COCO.json",
            "COCO",
            pytest.raises(ValueError),
        ),
    ],
)
def test_from_single_file_no_category(
    input_file: str,
    format: Literal["VIA", "COCO"],
    expected_exception: pytest.raises,
    annotations_test_data: dict,
):
    """Test the specific bounding box format readers when the input file
    has annotations with no category.
    """
    # Compute bboxes dataframe with input file that has no categories
    # (this should raise an error for COCO files)
    with expected_exception as excinfo:
        df = _from_single_file(
            file_path=annotations_test_data[input_file],
            format=format,
        )

    # If no error expected, check that the dataframe has empty categories
    if not excinfo:
        assert all(df.loc[:, "category"] == "")
        assert all(df.loc[:, "supercategory"] == "")


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
    # Note that "image_width" and "image_height" are not exported to the
    # VIA file
    for row in rows:
        assert all(
            key in row
            for key in [STANDARD_BBOXES_DF_INDEX] + STANDARD_BBOXES_DF_COLUMNS
            if key not in ["image_width", "image_height"]
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
    "duplicates_kwargs, expected_exception",
    [
        ({"ignore_index": True}, pytest.raises(ValueError)),
        ({"inplace": True}, pytest.raises(ValueError)),
        ({"subset": "image_id"}, does_not_raise()),
        ({"keep": "last"}, does_not_raise()),
    ],
)
@pytest.mark.parametrize(
    "input_format, filename",
    [
        ("VIA", "small_bboxes_duplicates_VIA.json"),
        ("VIA", "MULTIPLE_VIA_FILES_WITH_DUPLICATES"),
        ("COCO", "small_bboxes_duplicates_COCO.json"),
        ("COCO", "MULTIPLE_COCO_FILES_WITH_DUPLICATES"),
    ],
)
def test_from_files_kwargs(
    input_format: Literal["VIA", "COCO"],
    filename: str | list[str],
    duplicates_kwargs: dict,
    expected_exception: pytest.raises,
    annotations_test_data: dict,
    multiple_files_duplicates: dict,
):
    """Test the behaviour of the `from_files` function when passing
    input files with duplicates (in single files or across files),
    and different keyword-arguments to the `drop_duplicates` method.
    """
    # Check kwargs behaviour when passing multiple files
    if "MULTIPLE" in filename:
        list_files = multiple_files_duplicates[input_format]["files"]
        input_files = [file["path"] for file in list_files]

        n_duplicates = multiple_files_duplicates[input_format]["duplicates"]
        n_total_annotations = sum(
            [file["n_annotations"] for file in list_files]
        )

        expected_n_annotations = n_total_annotations - n_duplicates
        expected_n_images = multiple_files_duplicates[input_format]["n_images"]
        expected_annots_per_image = None

    # Check kwargs behaviour when passing a single file
    else:
        input_files = annotations_test_data[filename]
        expected_n_annotations = 3
        expected_n_images = 3
        expected_annots_per_image = 1

    # Compute dataframe and check if an error is raised
    with expected_exception as excinfo:
        df = from_files(
            input_files,
            format=input_format,
            **duplicates_kwargs,
        )
    if excinfo:
        assert (
            "argument for `pandas.DataFrame.drop_duplicates` "
            "may not be overridden." in str(excinfo.value)
        )

    # If no error expected: check dataframe content
    if expected_exception == does_not_raise():
        assert_dataframe(
            df,
            expected_n_annotations=expected_n_annotations,
            expected_n_images=expected_n_images,
            expected_supercategories="animal",
            expected_categories="crab",
            expected_annots_per_image=expected_annots_per_image,
        )
