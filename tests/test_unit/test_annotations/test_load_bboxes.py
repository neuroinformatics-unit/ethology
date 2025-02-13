import json
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pandas as pd
import pytest

from ethology.annotations.io.load_bboxes import (
    STANDARD_BBOXES_DF_COLUMNS,
    STANDARD_BBOXES_DF_INDEX,
    _df_rows_from_valid_COCO_file,
    _df_rows_from_valid_VIA_file,
    _from_multiple_files,
    _from_single_file,
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
            "ethology.annotations.io.load_bboxes._from_single_file",
        ),
        (
            [Path("/path/to/file1"), Path("/path/to/file2")],  # multiple files
            "ethology.annotations.io.load_bboxes._from_multiple_files",
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
    metadata is added correctly.
    """
    # Call general function and see if mocked function is called
    with patch(function_to_mock) as mock:
        df = from_files(file_path, format=format, images_dirs=images_dirs)
        mock.assert_called_once_with(file_path, format=format)

    # Check metadata
    assert df.attrs["annotation_files"] == file_path
    assert df.attrs["annotation_format"] == format
    if images_dirs:
        assert df.attrs["images_directories"] == images_dirs


@pytest.mark.parametrize(
    "format",
    [
        "VIA",
        "COCO",
    ],
)
def test_from_multiple_files(
    format: Literal["VIA", "COCO"], multiple_files: dict
):
    """Test that the general bounding boxes loading function reads
    correctly multiple files of the supported formats without any
    duplicates.
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
    df_all = _from_multiple_files(list_paths, format=format)

    # Check dataframe
    assert_dataframe(
        df_all,
        expected_n_annotations=n_annotations,
        expected_n_images=n_images,
        expected_supercategories="animal",
        expected_categories="crab",
    )


def test_from_single_file_unsupported():
    """Test that unsupported formats throw the expected errors."""
    file_path = Path("/mock/path/to/file")
    format = "unsupported"

    with pytest.raises(ValueError) as excinfo:
        _from_single_file(file_path=file_path, format=format)
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
def test_from_single_file(
    input_file: str,
    format: Literal["VIA", "COCO"],
    annotations_test_data: dict,
):
    """Test the specific bounding box format readers."""
    # Compute bboxes dataframe from a single file
    df = _from_single_file(
        file_path=annotations_test_data[input_file],
        format=format,
    )

    # Compute number of annotations and images in input file
    expected_n_images, expected_n_annotations = (
        count_imgs_and_annots_in_input_file(
            file_path=annotations_test_data[input_file],
            format=format,
        )
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
def test_from_single_file_duplicates(
    format: Literal["VIA", "COCO"],
    annotations_test_data: dict,
):
    """Test the specific bounding box format readers when the input file
    contains duplicate annotations.
    """
    # Properties of input data
    # in the "small_bboxes_duplicates_" files, one annotation is duplicated
    # in the first frame
    filepath = annotations_test_data[f"small_bboxes_duplicates_{format}.json"]
    expected_n_images, expected_n_annotations_w_duplicates = (
        count_imgs_and_annots_in_input_file(filepath, format=format)
    )
    n_duplicates = 1

    # Compute bboxes dataframe
    df = _from_single_file(
        file_path=annotations_test_data[
            f"small_bboxes_duplicates_{format}.json"
        ],
        format=format,
    )

    # Check dataframe has no duplicates
    assert_dataframe(
        df,
        expected_n_annotations=expected_n_annotations_w_duplicates
        - n_duplicates,
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
    """Test the specific bounding box format readers when the input file
    has annotations with no category.
    """
    # Compute bboxes dataframe with input file that has no categories
    # (this should raise an error for COCO files)
    with expected_exception as excinfo:
        df = _from_single_file(
            file_path=annotations_test_data[
                f"small_bboxes_no_cat_{format}.json"
            ],
            format=format,
        )

    # Check that the error message is as expected
    if excinfo:
        assert (
            "Empty value(s) found for the required key(s) "
            "['annotations', 'categories']." in str(excinfo.value)
        )
    # If no error expected, check that the dataframe has empty categories
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
    """Test the extraction of rows from a valid input file."""
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
    # Get expected size of dataframe when passing multiple files with
    # duplicates
    if "MULTIPLE" in input_file:
        # Get input data
        input_files = multiple_files_duplicates[format]["files"]
        n_duplicates = multiple_files_duplicates[format]["duplicates"]
        n_unique_images = multiple_files_duplicates[format]["n_images"]

        # Compute number of annotations, with and without duplicates
        n_total_annotations = sum(
            [
                count_imgs_and_annots_in_input_file(file, format)[1]
                for file in input_files
            ]
        )
        n_unique_annotations = n_total_annotations - n_duplicates
        expected_annots_per_image = None

    # Get expected size of dataframe when passing a single file with duplicates
    else:
        input_files = annotations_test_data[input_file]
        n_duplicates = 1
        n_unique_images, n_total_annotations = (
            count_imgs_and_annots_in_input_file(input_files, format)
        )
        n_unique_annotations = n_total_annotations - n_duplicates
        expected_annots_per_image = 1

    # Compute dataframe
    df = from_files(input_files, format=format)

    # Check dataframe content is as expected
    assert_dataframe(
        df,
        expected_n_annotations=n_unique_annotations,
        expected_n_images=n_unique_images,
        expected_supercategories="animal",
        expected_categories="crab",
        expected_annots_per_image=expected_annots_per_image,
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
    """Test if the bboxes dataframe image_id is assigned based on the
    alphabetically sorted list of filenames.
    """
    # Get path to file
    filepath = annotations_test_data[f"small_bboxes_image_id_{format}.json"]

    # Read data
    df = from_files(filepath, format=format)

    # Get image_id and filename pairs from the input data file
    with open(filepath) as f:
        data = json.load(f)

    # Compute expected image ID - filename pairs if ID computed alphabetically
    pairs_img_id_to_filename_alphabetical = {
        id: file
        for id, file in enumerate(sorted(df["image_filename"].tolist()))
    }

    # Check image ID in input data file is not assigned alphabetically
    if format == "VIA":
        list_via_images = data["_via_image_id_list"]
        pairs_img_id_to_filename_in = {
            list_via_images.index(img_via_ky): img_dict["filename"]
            for img_via_ky, img_dict in data["_via_img_metadata"].items()
        }
    elif format == "COCO":
        pairs_img_id_to_filename_in = {
            img_dict["id"]: img_dict["file_name"]
            for img_dict in data["images"]
        }
    assert pairs_img_id_to_filename_in != pairs_img_id_to_filename_alphabetical

    # Check image_id in dataframe is assigned alphabetically
    pairs_img_id_filename_out = {
        id: file
        for file, id in zip(df["image_filename"], df["image_id"], strict=True)
    }
    assert pairs_img_id_filename_out == pairs_img_id_to_filename_alphabetical


def test_dataframe_from_same_annotations(annotations_test_data: dict):
    """Test whether the same annotations exported to VIA and COCO formats
    produce the same dataframe, except for the image width and height columns.

    We use the `_subset` files because we know they contain the
    same annotations.
    """
    # Read data into dataframes
    df_via = from_files(
        annotations_test_data["small_bboxes_VIA_subset.json"],
        format="VIA",
    )
    df_coco = from_files(
        annotations_test_data["small_bboxes_COCO_subset.json"],
        format="COCO",
    )

    # Compare dataframes excluding `image_width`, `image_height` columns
    assert df_via.drop(columns=["image_width", "image_height"]).equals(
        df_coco.drop(columns=["image_width", "image_height"])
    )
