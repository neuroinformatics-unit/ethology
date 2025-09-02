import json
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ethology.io.annotations.load_bboxes import (
    STANDARD_BBOXES_DF_COLUMNS,
    STANDARD_BBOXES_DF_INDEX,
    _df_from_multiple_files,
    _df_from_single_file,
    _df_rows_from_valid_COCO_file,
    _df_rows_from_valid_VIA_file,
    from_files,
)


def count_imgs_and_annots_in_input_file(
    file_path: Path, format: Literal["VIA", "COCO"]
) -> tuple[int, int]:
    """Compute the number of images and annotations in the input file.

    Note that this function does not check for duplicates, so all
    counts are including any possible duplicates.
    """
    with open(file_path) as f:
        data = json.load(f)

    if format == "VIA":
        n_images = len(data["_via_img_metadata"])
        n_annotations = sum(
            len(img_dict["regions"]) if "regions" in img_dict else 0
            for img_dict in data["_via_img_metadata"].values()
        )
    elif format == "COCO":
        n_images = len(data["images"])
        n_annotations = len(data["annotations"])
    else:
        raise ValueError("Unsupported format")

    return n_images, n_annotations


def get_list_images(
    file_path: Path | list[Path], format: Literal["VIA", "COCO"]
) -> list:
    """Extract list of image files from input annotation file."""
    # Read input data as dict
    list_data = []
    if isinstance(file_path, list):
        for file in file_path:
            with open(file) as f:
                list_data.append(json.load(f))
    else:
        with open(file_path) as f:
            list_data.append(json.load(f))

    # Extract list of images ordered as in the input data
    if format == "VIA":
        return [
            img["filename"]
            for data in list_data
            for img in data["_via_img_metadata"].values()
        ]
    elif format == "COCO":
        return [
            img["file_name"] for data in list_data for img in data["images"]
        ]
    else:
        raise ValueError("Unsupported format")


@pytest.fixture
def multiple_files(annotations_test_data: dict) -> dict:
    """Fixture that returns for each format, a pair of annotation files."""
    return {
        "VIA": [
            annotations_test_data["VIA_JSON_sample_1.json"],
            annotations_test_data["VIA_JSON_sample_2.json"],
        ],
        "COCO": [
            annotations_test_data["COCO_JSON_sample_1.json"],
            annotations_test_data["COCO_JSON_sample_2.json"],
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
                annotations_test_data["small_bboxes_VIA.json"],
                annotations_test_data[
                    "small_bboxes_VIA_subset.json"
                ],  # both annotations appear in "small_bboxes_VIA.json" too
            ],
            "duplicates": 2,
            "n_images": 3,
            "max_annots_per_image": 1,
        },
        "COCO": {
            "files": [
                annotations_test_data["small_bboxes_COCO.json"],
                annotations_test_data["small_bboxes_COCO_subset_plus.json"],
                # two annotations appear in "small_bboxes_COCO.json" too,
                # the third one is new
            ],
            "duplicates": 2,
            "n_images": 3,
            "max_annots_per_image": 2,
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


def assert_dataset(
    ds: xr.Dataset,
    expected_n_images: int,
    expected_n_annotations: int,
    expected_max_annots_per_image: int,
    expected_space_dim: int,
    expected_n_categories: int,
    expected_category_str: list[str],
):
    """Check that the dataset has the expected shape and content."""
    # Check shape of position array
    assert ds.position.shape == (
        expected_n_images,
        expected_space_dim,
        expected_max_annots_per_image,
    )

    # Check shape of category array
    assert ds.category.shape == (
        expected_n_images,
        expected_max_annots_per_image,
    )

    # Check total number of no nan annotations
    assert (
        np.sum(np.any(~np.isnan(ds.position.values), axis=1))
        == expected_n_annotations
    )

    # Check total number of non -1 categories
    assert (
        np.sum(np.unique(ds.category.values.flatten()) != -1)
        == expected_n_categories
    )

    # Check map from category_id to category name is as expected
    assert sorted(ds.attrs["map_category_to_str"].values()) == sorted(
        expected_category_str
    )


@pytest.mark.parametrize(
    "format",
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
            "ethology.io.annotations.load_bboxes._df_from_single_file",
        ),
        (
            [Path("/path/to/file1"), Path("/path/to/file2")],  # multiple files
            "ethology.io.annotations.load_bboxes._df_from_multiple_files",
        ),
    ],
)
def test_from_files(
    format: Literal["VIA", "COCO"],
    images_dirs: Path | str | list[Path | str] | None,
    file_path: Path,
    function_to_mock: str,
):
    """Test that the general bounding boxes loading function delegates
    correctly to the single or multiple file readers, and check the
    metadata is added correctly to the xarray dataset.
    """
    # Create a mock intermediate DataFrame with one image and one annotation,
    # to return from the mocked reader function
    mock_df_all = pd.DataFrame(
        {
            "image_filename": ["test_image.jpg"],
            "image_id": [0],
            "x_min": [10.0],
            "y_min": [20.0],
            "width": [100.0],
            "height": [200.0],
            "supercategory": ["animal"],
            "category": ["crab"],
            "category_id": [1],
            "image_width": [800],
            "image_height": [600],
        }
    )

    # Call general function and see if mocked function is called
    with patch(function_to_mock, return_value=mock_df_all) as mock:
        ds = from_files(file_path, format=format, images_dirs=images_dirs)
        mock.assert_called_once_with(file_path, format=format)

    # Check metadata
    assert ds.attrs["annotation_files"] == file_path
    assert ds.attrs["annotation_format"] == format
    if images_dirs:
        assert ds.attrs["images_directories"] == images_dirs

    # Check that the maps exist and are not empty
    assert ds.attrs["map_category_to_str"] == {1: "crab"}
    assert ds.attrs["map_image_id_to_filename"] == {0: "test_image.jpg"}


@pytest.mark.parametrize(
    "format",
    [
        "VIA",
        "COCO",
    ],
)
def test_df_from_multiple_files(
    format: Literal["VIA", "COCO"], multiple_files: dict
):
    """Test that the multiple files reader reads correctly multiple files
    of the supported formats without any duplicates as a dataframe.
    """
    # Get list of paths
    list_paths = multiple_files[format]

    # Compute total number of annotations and images
    n_images = sum(
        count_imgs_and_annots_in_input_file(file, format=format)[0]
        for file in list_paths
    )
    n_annotations = sum(
        count_imgs_and_annots_in_input_file(file, format=format)[1]
        for file in list_paths
    )

    # Read all files as a dataframe
    df_all = _df_from_multiple_files(list_paths, format=format)

    # Check dataframe
    assert_dataframe(
        df_all,
        expected_n_annotations=n_annotations,
        expected_n_images=n_images,
        expected_supercategories="animal",
        expected_categories="crab",
    )


def test_df_from_single_file_unsupported():
    """Test that the single file reader throws the expected errors
    for unsupported formats.
    """
    file_path = Path("/mock/path/to/file")
    format = "unsupported"
    with pytest.raises(ValueError) as excinfo:
        _df_from_single_file(file_path=file_path, format=format)

    assert "Unsupported format" in str(excinfo.value)


@pytest.mark.parametrize(
    ("input_file, format"),
    [
        ("VIA_JSON_sample_1.json", "VIA"),  # medium VIA file
        ("VIA_JSON_sample_2.json", "VIA"),  # medium VIA file
        ("small_bboxes_VIA.json", "VIA"),  # small VIA file
        ("COCO_JSON_sample_1.json", "COCO"),  # medium COCO file
        ("COCO_JSON_sample_2.json", "COCO"),  # medium COCO file
        ("small_bboxes_COCO.json", "COCO"),  # small COCO file
    ],
)
def test_df_from_single_file(
    input_file: str,
    format: Literal["VIA", "COCO"],
    annotations_test_data: dict,
):
    """Test the single file reader reads correctly a single file
    of the supported formats as a dataframe.
    """
    # Compute number of annotations and images in input file
    expected_n_images, expected_n_annotations = (
        count_imgs_and_annots_in_input_file(
            file_path=annotations_test_data[input_file],
            format=format,
        )
    )

    # Compute bboxes dataframe from a single file
    df = _df_from_single_file(
        file_path=annotations_test_data[input_file],
        format=format,
    )

    # Check dataframe
    # (we only check annotations per image for the small datasets)
    assert_dataframe(
        df,
        expected_n_annotations,
        expected_n_images,
        expected_supercategories="animal",
        expected_categories="crab",
        expected_annots_per_image=1 if expected_n_images < 5 else None,
    )


@pytest.mark.parametrize(
    ("format"),
    [
        "VIA",
        "COCO",
    ],
)
def test_df_from_single_file_duplicates(
    format: Literal["VIA", "COCO"],
    annotations_test_data: dict,
):
    """Test the single file reader reads correctly a single file
    of the supported formats as a dataframe when the input file
    contains duplicate annotations.
    """
    # In the "small_bboxes_duplicates_" files, one annotation is duplicated
    # in the first frame
    filepath = annotations_test_data[f"small_bboxes_duplicates_{format}.json"]
    expected_n_images, n_annotations_with_duplicates = (
        count_imgs_and_annots_in_input_file(filepath, format=format)
    )
    n_duplicates = 1
    expected_n_annotations = n_annotations_with_duplicates - n_duplicates

    # Compute bboxes dataframe
    df = _df_from_single_file(
        file_path=filepath,
        format=format,
    )

    # Check dataframe has no duplicates
    assert_dataframe(
        df,
        expected_n_annotations=expected_n_annotations,
        expected_n_images=expected_n_images,
        expected_supercategories="animal",
        expected_categories="crab",
    )


@pytest.mark.parametrize(
    ("format, expected_exception"),
    [
        (
            "VIA",
            does_not_raise(),
        ),
        (
            "COCO",
            pytest.raises(ValueError),
        ),
    ],
)
def test_from_single_file_no_category(
    format: Literal["VIA", "COCO"],
    expected_exception: pytest.raises,
    annotations_test_data: dict,
):
    """Test the single file reader reads correctly a single file
    of the supported formats as a dataframe when the input file
    has annotations with no category.
    """
    # Compute bboxes dataframe with input file that has no categories
    # (this should raise an error for COCO files)
    with expected_exception as excinfo:
        filepath = annotations_test_data[f"small_bboxes_no_cat_{format}.json"]
        df = _df_from_single_file(file_path=filepath, format=format)

    # Check that the error message is as expected
    if excinfo:
        assert (
            "Empty value(s) found for the required key(s) "
            "['annotations', 'categories']." in str(excinfo.value)
        )
    # If no error expected (i.e. for VIA files), check that the dataframe
    # has empty category columns
    else:
        assert df["category"].isna().all()
        assert df["supercategory"].isna().all()
        assert df["category_id"].isna().all()


@pytest.mark.parametrize(
    "input_file, format",
    [
        ("VIA_JSON_sample_1.json", "VIA"),
        ("VIA_JSON_sample_2.json", "VIA"),
        ("small_bboxes_VIA.json", "VIA"),
        ("small_bboxes_duplicates_VIA.json", "VIA"),  # contains duplicates
        ("COCO_JSON_sample_1.json", "COCO"),
        ("COCO_JSON_sample_2.json", "COCO"),
        ("small_bboxes_COCO.json", "COCO"),
        ("small_bboxes_duplicates_COCO.json", "COCO"),  # contains duplicates
    ],
)
def test_df_rows_from_valid_file(
    input_file: str,
    format: Literal["VIA", "COCO"],
    annotations_test_data: dict,
):
    """Test the extraction of dataframe rows from a valid input file."""
    # Determine row function to test
    if format == "VIA":
        row_function_to_test = _df_rows_from_valid_VIA_file
    elif format == "COCO":
        row_function_to_test = _df_rows_from_valid_COCO_file
    else:
        raise ValueError("Unsupported format")

    # Extract rows from file
    filepath = annotations_test_data[input_file]
    rows = row_function_to_test(filepath)

    # Check there are as many rows as annotations
    _, expected_n_annotations = count_imgs_and_annots_in_input_file(
        filepath, format=format
    )
    assert len(rows) == expected_n_annotations

    # Check each row contains required column data
    # Note that "image_width" and "image_height" are not defined in the
    # VIA file, so we exclude them from the required keys.
    required_keys = [STANDARD_BBOXES_DF_INDEX] + STANDARD_BBOXES_DF_COLUMNS
    if format == "VIA":
        required_keys = [
            key
            for key in required_keys
            if key not in ["image_width", "image_height"]
        ]
    assert all([key in row for key in required_keys for row in rows])


@pytest.mark.parametrize(
    "input_file, format",
    [
        (
            "small_bboxes_duplicates_VIA.json",
            "VIA",
        ),  # one annotation is duplicated in the first frame
        (
            "MULTIPLE_VIA_FILES_WITH_DUPLICATES",
            "VIA",
        ),
        (
            "small_bboxes_duplicates_COCO.json",
            "COCO",
        ),  # one annotation is duplicated in the first frame
        (
            "MULTIPLE_COCO_FILES_WITH_DUPLICATES",
            "COCO",
        ),
    ],
)
def test_from_files_duplicates(
    input_file: str | list[str],
    format: Literal["VIA", "COCO"],
    annotations_test_data: dict,
    multiple_files_duplicates: dict,
):
    """Test the behaviour of the `from_files` function when passing
    input files with duplicates (in single files or across files).
    """
    # Get expected size of dataset when passing multiple files with
    # duplicates
    if "MULTIPLE" in input_file:
        # Get input data
        input_files = multiple_files_duplicates[format]["files"]
        n_duplicates = multiple_files_duplicates[format]["duplicates"]
        n_unique_images = multiple_files_duplicates[format]["n_images"]
        max_annots_per_image = multiple_files_duplicates[format][
            "max_annots_per_image"
        ]

        # Compute number of annotations, with and without duplicates
        n_total_annotations = sum(
            [
                count_imgs_and_annots_in_input_file(file, format)[1]
                for file in input_files
            ]
        )
        n_unique_annotations = n_total_annotations - n_duplicates

    # Get expected size of dataframe when passing a single file with duplicates
    else:
        input_files = annotations_test_data[input_file]
        n_duplicates = 1
        n_unique_images, n_total_annotations = (
            count_imgs_and_annots_in_input_file(input_files, format)
        )
        n_unique_annotations = n_total_annotations - n_duplicates
        max_annots_per_image = 1

    # Compute dataset
    ds = from_files(input_files, format=format)

    # Check dataset content is as expected
    assert_dataset(
        ds,
        expected_n_images=n_unique_images,
        expected_n_annotations=n_unique_annotations,
        expected_max_annots_per_image=max_annots_per_image,
        expected_space_dim=2,
        expected_n_categories=1,
        expected_category_str=["crab"],
    )


@pytest.mark.parametrize(
    "format",
    [
        "VIA",
        "COCO",
    ],
)
def test_image_id_assignment(
    format: Literal["VIA", "COCO"], annotations_test_data: dict
):
    """Test that the bboxes dataset "image_id" is assigned based on the
    alphabetically sorted list of image filenames.
    """
    # Read data
    filepath = annotations_test_data[f"small_bboxes_image_id_{format}.json"]
    ds = from_files(filepath, format=format)

    # Read input data file as a dictionary
    with open(filepath) as f:
        data = json.load(f)

    # Compute expected map from image ID to filename if image ID assigned
    # alphabetically
    sorted_filenames = sorted(ds.attrs["map_image_id_to_filename"].values())
    map_img_id_to_filename_alphabetical = dict(enumerate(sorted_filenames))

    # Check image ID in input data file is not assigned alphabetically
    if format == "VIA":
        list_via_images = data["_via_image_id_list"]
        map_img_id_to_filename_in = {
            list_via_images.index(img_via_ky): img_dict["filename"]
            for img_via_ky, img_dict in data["_via_img_metadata"].items()
        }
    elif format == "COCO":
        map_img_id_to_filename_in = {
            img_dict["id"]: img_dict["file_name"]
            for img_dict in data["images"]
        }
    assert map_img_id_to_filename_in != map_img_id_to_filename_alphabetical

    # Check image_id in output dataset is assigned alphabetically
    assert (
        ds.attrs["map_image_id_to_filename"]
        == map_img_id_to_filename_alphabetical
    )


def test_dataset_from_same_annotations(annotations_test_data: dict):
    """Test that the same annotations exported to VIA and COCO formats
    produce the same dataset, except for the image width and height columns.

    We use the `_subset.json` test files because we know they contain the
    same annotations.
    """
    # Read data into dataframes
    ds_via = from_files(
        annotations_test_data["small_bboxes_VIA_subset.json"],
        format="VIA",
    )
    ds_coco = from_files(
        annotations_test_data["small_bboxes_COCO_subset.json"],
        format="COCO",
    )

    # Compare datasets ignoring datasetattributes
    # Two datasets are equal if they have matching variables and coordinates
    assert ds_via.equals(ds_coco)

    # Check attributes individually
    assert (
        ds_via.attrs["annotation_files"] != ds_coco.attrs["annotation_files"]
    )
    assert (
        ds_via.attrs["annotation_format"] != ds_coco.attrs["annotation_format"]
    )
    assert (
        ds_via.attrs["map_category_to_str"]
        == ds_coco.attrs["map_category_to_str"]
    )
    assert (
        ds_via.attrs["map_image_id_to_filename"]
        == ds_coco.attrs["map_image_id_to_filename"]
    )
    assert (
        ds_via.attrs["images_directories"]
        == ds_coco.attrs["images_directories"]
    )


@pytest.mark.parametrize(
    "input_file, format, case_category_id, expected_category_id",
    [
        (
            "small_bboxes_no_cat_VIA.json",
            "VIA",
            "empty",
            None,
        ),  # no category in VIA file --> should be None in df
        (
            "small_bboxes_VIA.json",
            "VIA",
            "string_integer",
            1,
        ),  # category ID is a string ("1") ---> should be 1 in df
        (
            "VIA_JSON_sample_1.json",
            "VIA",
            "string_category",
            1,
        ),  # category ID is a string ("crab") ---> should be 1 in df
        (
            "small_bboxes_COCO.json",
            "COCO",
            "integer",
            1,
        ),  # category ID is an integer (1) ---> should be 1 in df
    ],
)
def test_category_id_extraction(
    input_file: str,
    format: Literal["VIA", "COCO"],
    case_category_id: str,
    expected_category_id: int | None,
    annotations_test_data: dict,
):
    """Test that the category_id is extracted correctly from the input file.

    VIA categories are saved as strings, while COCO categories are saved as
    integers. Note that COCO category IDs are always 1-based indices, and
    maintained when read into the ethology dataframe (0 is reserved for the
    background class).
    """
    df = _df_from_single_file(
        file_path=annotations_test_data[input_file], format=format
    )

    # If no category in input file, the category_id column should be None
    if case_category_id == "empty":
        df["category_id"].apply(lambda x: x is expected_category_id).all()

    # Category ID should be an integer
    elif case_category_id in ["string_integer", "integer", "string_category"]:
        assert df["category_id"].dtype == int
        assert df["category_id"].unique() == [expected_category_id]

    # Category ID should be factorized into 1-based integers if saved as
    # strings
    elif case_category_id == "string_category":
        assert all(df["category_id"] == df["category"].factorize()[0] + 1)


@pytest.mark.parametrize(
    "input_file_type, format",
    [
        ("single", "COCO"),
        ("multiple", "COCO"),
        ("multiple", "VIA"),
    ],
)
def test_annotations_sorted_by_image_filename(
    input_file_type: str,
    format: Literal["VIA", "COCO"],
    annotations_test_data: dict,
):
    """Test that the annotations in the dataset are sorted by image filename.

    We use the `small_bboxes_image_id_COCO.json` data because the list of
    image filenames in them is not sorted.

    We don't test the case of a single VIA file because the VIA tool always
    outputs the image dictionaries under `_via_img_metadata` sorted by
    filename.
    """
    # Get input file(s)
    if input_file_type == "single":
        input_filepaths = annotations_test_data[
            f"small_bboxes_image_id_{format}.json"
        ]
    elif input_file_type == "multiple":
        input_files = (
            ["VIA_JSON_sample_2.json", "VIA_JSON_sample_1.json"]
            if format == "VIA"
            else ["COCO_JSON_sample_1.json", "COCO_JSON_sample_2.json"]
        )  # load in this order so that the image filenames are not sorted
        input_filepaths = [annotations_test_data[file] for file in input_files]

    # Check list of images as they are in the input data are unsorted
    list_input_images = get_list_images(input_filepaths, format=format)
    assert list_input_images != sorted(list_input_images)

    # Compute annotations dataset
    ds = from_files(file_paths=input_filepaths, format=format)

    # Check that the image_id to filename map in the dataset is sorted
    sorted_filenames = sorted(ds.attrs["map_image_id_to_filename"].values())
    map_img_id_to_filename_sorted = dict(enumerate(sorted_filenames))
    assert (
        ds.attrs["map_image_id_to_filename"] == map_img_id_to_filename_sorted
    )
