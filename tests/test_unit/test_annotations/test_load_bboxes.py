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

    # Compare dataframes excluding `image_width`, `image_height` and
    # `category_id` columns
    assert df_via.drop(
        columns=["image_width", "image_height", "category_id"]
    ).equals(
        df_coco.drop(columns=["image_width", "image_height", "category_id"])
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
        # VIA category IDs are retained
        (
            "VIA_JSON_sample_1.json",
            "VIA",
            "string_category",
            0,
        ),  # category ID is a string ("crab") ---> should be factorized
        (
            "small_bboxes_COCO.json",
            "COCO",
            "integer",
            0,
        ),  # category ID is an integer (1) ---> should be 0 in df
        # COCO category IDs are always 1-based indices, and transformed to
        # 0-based indices when read into df
    ],
)
def test_category_id_extraction(
    input_file: str,
    format: Literal["VIA", "COCO"],
    case_category_id: str,
    expected_category_id: int | None,
    annotations_test_data: dict,
):
    """Test that the category_id is extracted correctly from the input file."""
    df = _from_single_file(
        file_path=annotations_test_data[input_file],
        format=format,
    )

    if case_category_id == "empty":
        df["category_id"].apply(lambda x: x is expected_category_id).all()

    elif case_category_id in ["string_integer", "integer"]:
        assert df["category_id"].dtype == int
        assert df["category_id"].unique() == [expected_category_id]

    elif case_category_id == "string_category":
        assert df["category_id"].dtype == int
        assert df["category_id"].unique() == [expected_category_id]
        assert all(df["category_id"] == df["category"].factorize()[0])


@pytest.mark.parametrize(
    "input_file_type, format",
    [
        ("multiple", "VIA"),
        ("single", "COCO"),
        ("multiple", "COCO"),
    ],
)
def test_sorted_annotations_by_image_filename(
    input_file_type: str,
    format: Literal["VIA", "COCO"],
    annotations_test_data: dict,
    # multiple_files: dict,
):
    """Test that the annotations are sorted by image filename.

    We use the `small_bboxes_image_id_COCO` data because the list of
    image filenames is not sorted.

    We don't test with a single VIA file because the VIA tool always
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

    # Compute bboxes dataframe
    df = from_files(file_paths=input_filepaths, format=format)

    # Check that the annotations in the dataframe are sorted by image filename
    assert df["image_filename"].to_list() == sorted(
        df["image_filename"].to_list()
    )


# --- TESTS FOR format="auto" ---


@pytest.mark.parametrize(
    ("input_file, format"),
    [
        ("small_bboxes_VIA.json", "VIA"),
        ("small_bboxes_COCO.json", "COCO"),
        ("VIA_JSON_sample_1.json", "VIA"),
        ("COCO_JSON_sample_1.json", "COCO"),
    ],
)
def test_from_files_auto_detect_single_success(
    input_file: str,
    format: Literal["VIA", "COCO"],
    annotations_test_data: dict,
):
    """Test `from_files` with format='auto' successfully detects format."""
    filepath = annotations_test_data[input_file]

    # Load with auto-detection
    df = from_files(file_paths=filepath, format="auto")

    # Check loaded data is correct (using existing helpers)
    expected_n_images, expected_n_annotations = (
        count_imgs_and_annots_in_input_file(filepath, format=format)
    )
    assert_dataframe(
        df,
        expected_n_annotations,
        expected_n_images,
        expected_supercategories="animal",  # Assuming test data uses this
        expected_categories="crab",  # Assuming test data uses this
        expected_annots_per_image=1 if "small" in input_file else None,
    )
    # Check the format stored in attributes is correct
    assert df.attrs["annotation_format"] == format


@pytest.mark.parametrize(
    "format",
    [
        "VIA",
        "COCO",
    ],
)
def test_from_files_auto_detect_multiple_success(
    format: Literal["VIA", "COCO"], multiple_files: dict
):
    """Test `from_files` with format='auto' detects format from the first
    file in a list and loads correctly, emitting a warning.
    """
    list_paths = multiple_files[format]

    # Compute expected total number of annotations and images
    n_images_total = sum(
        count_imgs_and_annots_in_input_file(file, format=format)[0]
        for file in list_paths
    )
    n_annotations_total = sum(
        count_imgs_and_annots_in_input_file(file, format=format)[1]
        for file in list_paths
    )

    # Check that the warning is emitted when loading multiple files with auto
    with pytest.warns(UserWarning) as record:
        df_all = from_files(list_paths, format="auto")

    # Check the warning message content (optional but good)
    assert len(record) == 1
    assert f"Format automatically detected as '{format}'" in str(
        record[0].message
    )
    assert "Assuming all files in the list have the same format" in str(
        record[0].message
    )

    # Check dataframe (adapt expected categories/supercategories if needed)
    assert_dataframe(
        df_all,
        expected_n_annotations=n_annotations_total,
        expected_n_images=n_images_total,
        expected_supercategories="animal",
        expected_categories="crab",
    )
    assert df_all.attrs["annotation_format"] == format


# --- Fixtures for creating invalid files for error testing ---


@pytest.fixture
def invalid_json_file(tmp_path: Path) -> Path:
    """Create a file that is not valid JSON."""
    p = tmp_path / "invalid.json"
    p.write_text("{invalid json structure", encoding="utf-8")
    return p


@pytest.fixture
def non_dict_json_file(tmp_path: Path) -> Path:
    """Create a file containing a valid JSON list, not a dict."""
    p = tmp_path / "list.json"
    p.write_text('["a", "b", "c"]', encoding="utf-8")
    return p


@pytest.fixture
def ambiguous_json_file(tmp_path: Path) -> Path:
    """Create a file with both VIA and COCO top-level keys."""
    p = tmp_path / "ambiguous.json"
    data: dict = {
        "_via_img_metadata": {},
        "_via_attributes": {},
        "images": [],
        "annotations": [],
        "categories": [],
    }
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


@pytest.fixture
def unknown_format_json_file(tmp_path: Path) -> Path:
    """Create a valid JSON dict file with unknown keys."""
    p = tmp_path / "unknown.json"
    p.write_text('{"other_key": 1, "another": "data"}', encoding="utf-8")
    return p


# --- Tests for format="auto" error conditions ---


def test_from_files_auto_detect_file_not_found():
    """Test `from_files` with format='auto' raises error if file not found."""
    non_existent_path = Path("./non_existent_file.json")
    assert not non_existent_path.exists()  # Ensure it doesn't exist

    with pytest.raises(ValueError) as excinfo:
        from_files(non_existent_path, format="auto")

    assert "Automatic format detection failed" in str(excinfo.value)
    assert "Annotation file not found" in str(excinfo.value.__cause__)


def test_from_files_auto_detect_invalid_json(invalid_json_file: Path):
    """Test `from_files` with format='auto' raises error for invalid JSON."""
    with pytest.raises(ValueError) as excinfo:
        from_files(invalid_json_file, format="auto")

    assert "Automatic format detection failed" in str(excinfo.value)
    assert "Error decoding JSON data" in str(excinfo.value.__cause__)


def test_from_files_auto_detect_non_dict_json(non_dict_json_file: Path):
    """Test `from_files` with format='auto' raises error for non-dict JSON."""
    with pytest.raises(ValueError) as excinfo:
        from_files(non_dict_json_file, format="auto")

    assert "Automatic format detection failed" in str(excinfo.value)
    assert "Expected JSON root to be a dictionary" in str(
        excinfo.value.__cause__
    )


def test_from_files_auto_detect_ambiguous_format(ambiguous_json_file: Path):
    """Test `from_files` with format='auto' raises error for ambiguous
    format.
    """
    with pytest.raises(ValueError) as excinfo:
        from_files(ambiguous_json_file, format="auto")

    assert "Automatic format detection failed" in str(excinfo.value)
    assert "contains keys characteristic of *both* VIA and COCO" in str(
        excinfo.value.__cause__
    )


def test_from_files_auto_detect_unknown_format(
    unknown_format_json_file: Path,
):
    """Test `from_files` with format='auto' raises error for unknown format."""
    with pytest.raises(ValueError) as excinfo:
        from_files(unknown_format_json_file, format="auto")

    assert "Automatic format detection failed" in str(excinfo.value)
    assert "Could not automatically determine format" in str(
        excinfo.value.__cause__
    )


def test_from_files_empty_list_input():
    """Test `from_files` raises ValueError for empty list input."""
    with pytest.raises(ValueError) as excinfo:
        from_files([])
    assert "Input 'file_paths' list cannot be empty" in str(excinfo.value)
