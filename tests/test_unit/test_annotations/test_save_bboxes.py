import json
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import pandas as pd
import pytest

from ethology.annotations.io.load_bboxes import from_files
from ethology.annotations.io.save_bboxes import (
    STANDARD_BBOXES_DF_COLUMNS_TO_COCO,
    _create_COCO_dict,
    _fill_in_COCO_required_data,
    _validate_df_bboxes,
    to_COCO_file,
)


def read_JSON_as_dict(file_path: str | Path) -> dict:
    """Read a JSON file and return its content as a dictionary."""
    with open(file_path) as file:
        return json.load(file)


def assert_list_of_dicts_match(
    dicts_1_unsorted: list,
    dicts_2_unsorted: list,
    key_to_sort_by: str,
    keys_to_exclude: list | None = None,
):
    """Assert two lists of dictionaries are equal after sorting by a key.

    Some keys can be excluded from the comparison via the `keys_to_exclude`
    parameter.
    """
    # Sort list of dictionaries
    list_dicts_1 = sorted(dicts_1_unsorted, key=lambda x: x[key_to_sort_by])
    list_dicts_2 = sorted(dicts_2_unsorted, key=lambda x: x[key_to_sort_by])

    # Prepare list of keys to exclude from comparison
    if keys_to_exclude is None:
        keys_to_exclude = []

    # Compare each dictionary in the lists
    for dict_1, dict_2 in zip(list_dicts_1, list_dicts_2, strict=True):
        # Extract common keys
        common_keys = set(dict_1.keys()).intersection(dict_2.keys())
        assert all(
            dict_1[ky] == dict_2[ky]
            for ky in common_keys
            if ky not in keys_to_exclude
        )


@pytest.fixture
def sample_bboxes_df() -> Callable:
    """Return a factory function for a sample bboxes dataframe.

    The factory function can be called with the `columns_to_drop` parameter
    """

    def _sample_bboxes_df_drop(
        columns_to_drop: list | None = None,
    ) -> pd.DataFrame:
        """Return a sample bboxes dataframe with the specified columns dropped.

        The original dataframe is from "small_bboxes_COCO.json" data with the
        relevant columns for COCO export added. The index is set to
        "annotation_id".
        """
        df = pd.DataFrame(
            {
                "annotation_id": {0: 0, 1: 1, 2: 2},
                "image_filename": {
                    0: "00000.jpg",
                    1: "00083.jpg",
                    2: "00166.jpg",
                },
                "image_id": {0: 0, 1: 83, 2: 166},
                "x_min": {0: 963, 1: 376, 2: 458},
                "y_min": {0: 283, 1: 314, 2: 329},
                "width": {0: 302, 1: 301, 2: 301},
                "height": {0: 172, 1: 123, 2: 131},
                "area": {0: 51944, 1: 37023, 2: 39431},
                "supercategory": {0: "animal", 1: "animal", 2: "animal"},
                "category": {0: "crab", 1: "crab", 2: "crab"},
                "category_id": {0: 1, 1: 1, 2: 1},
                "iscrowd": {0: 0, 1: 0, 2: 0},
                "segmentation": {
                    0: [[963, 283, 1265, 283, 1265, 455, 963, 455]],
                    1: [[376, 314, 677, 314, 677, 437, 376, 437]],
                    2: [[458, 329, 759, 329, 759, 460, 458, 460]],
                },
                "bbox": {
                    0: [963, 283, 302, 172],
                    1: [376, 314, 301, 123],
                    2: [458, 329, 301, 131],
                },
                "image_width": {0: 1280, 1: 1280, 2: 1280},
                "image_height": {0: 720, 1: 720, 2: 720},
            }
        ).set_index("annotation_id")

        # Drop columns if specified
        if columns_to_drop:
            return df.drop(columns=columns_to_drop)
        else:
            return df

    return _sample_bboxes_df_drop


@pytest.mark.parametrize(
    "df, expected_exception, expected_error_message",
    [
        (
            [],
            pytest.raises(TypeError),
            "Expected a pandas DataFrame, but got <class 'list'>.",
        ),
        (
            pd.DataFrame(),
            pytest.raises(ValueError),
            "Expected index name to be 'annotation_id', but got 'None'.",
        ),
        (
            pd.DataFrame({"annotation_id": [1, 2, 3]}).set_index(
                "annotation_id"
            ),
            pytest.raises(ValueError),
            "Required bounding box coordinates "
            "'x_min', 'y_min', 'width', 'height', are not present in "
            "the dataframe.",
        ),
        (
            "sample_bboxes_df",
            does_not_raise(),
            "",
        ),
    ],
)
def test_validate_df_bboxes(
    df: pd.DataFrame,
    expected_exception: pytest.raises,
    expected_error_message: str,
    request: pytest.FixtureRequest,
):
    """Test the validation function for the bboxes dataframe throws the
    expected errors.
    """
    if not expected_error_message:
        df_factory = request.getfixturevalue(df)
        df = df_factory()
    with expected_exception as excinfo:
        _validate_df_bboxes(df)
    if excinfo:
        assert expected_error_message == str(excinfo.value)


@pytest.mark.parametrize(
    "columns_to_drop",
    [
        [],
        ["category"],
        ["category_id"],
        ["area"],
        ["iscrowd"],
        ["segmentation"],
        ["bbox"],
    ],
)
def test_fill_in_COCO_required_data(
    columns_to_drop: list, sample_bboxes_df: Callable
):
    """Test the fill-in function for exporting to COCO fills in any
    columns required by COCO that may be missing.
    """
    # Get dataframe with dropped columns
    df_full = sample_bboxes_df()
    df_input = sample_bboxes_df(columns_to_drop=columns_to_drop)

    # Check relevant column is missing from the input
    if columns_to_drop:
        assert all(x not in df_input.columns for x in columns_to_drop)

    # Fill in missing columns
    df_output = _fill_in_COCO_required_data(df_input.copy())

    # Check
    assert df_output.index.name == "annotation_id"
    assert all(
        x in df_output.columns
        for x in [
            "category",
            "category_id",
            "area",
            "iscrowd",
            "segmentation",
            "bbox",
        ]
    )

    # Check fill in values are as original
    if columns_to_drop == ["category"]:
        assert all(df_output["category"] == "")
    elif columns_to_drop == ["category_id"]:
        assert all(
            df_output["category_id"]
            == df_full["category"].factorize(sort=True)[0]
        )
    else:
        assert df_full[columns_to_drop].equals(df_output[columns_to_drop])


def test_create_COCO_dict(sample_bboxes_df: Callable):
    """Test that the function that transforms the modified bboxes dataframe to
    a COCO dictionary creates a dictionary as expected.
    """
    # Prepare input data
    df = sample_bboxes_df()
    df["annotation_id"] = df.index  # required to extract the COCO dict

    # Extract COCO dictionary
    COCO_dict = _create_COCO_dict(df)

    # Check type and sections
    assert isinstance(COCO_dict, dict)
    assert all(x in COCO_dict for x in ["images", "categories", "annotations"])

    # Check keys in each section
    for section, section_mapping in STANDARD_BBOXES_DF_COLUMNS_TO_COCO.items():
        assert all(
            [
                x in elem
                for elem in COCO_dict[section]
                for x in section_mapping.values()
            ]
        )


@pytest.mark.parametrize(
    "filename",
    [
        "small_bboxes_COCO.json",
        pytest.param(
            "COCO_JSON_sample_1.json",
            marks=pytest.mark.xfail(reason="should pass after PR48"),
        ),
    ],
)
def test_df_bboxes_to_COCO_file(
    filename: str, annotations_test_data: dict, tmp_path: Path
):
    # Get input JSON file
    input_file = annotations_test_data[filename]

    # Read as bboxes dataframe
    df = from_files(input_file, format="COCO")

    # Export dataframe to COCO format
    output_file = to_COCO_file(df, output_filepath=tmp_path / "output.json")

    input_dict = read_JSON_as_dict(input_file)
    output_dict = read_JSON_as_dict(output_file)

    # Check lists of "categories" dictionaries match
    assert_list_of_dicts_match(
        input_dict["categories"],
        output_dict["categories"],
        key_to_sort_by="id",
        keys_to_exclude=["id"],  # "id" is expected to be different
    )

    # Check lists of "images" dictionaries match
    assert_list_of_dicts_match(
        input_dict["images"],
        output_dict["images"],
        key_to_sort_by="file_name",
        keys_to_exclude=None,
    )

    # Check lists of "annotations" dictionaries match
    assert_list_of_dicts_match(
        input_dict["annotations"],
        output_dict["annotations"],
        key_to_sort_by="id",
        keys_to_exclude=["id", "category_id"],
    )

    # Check category_id is as expected for COCO files exported with VIA tool
    # under categories
    assert all(
        categories_out["id"] == categories_in["id"] - 1
        for categories_in, categories_out in zip(
            input_dict["categories"],
            output_dict["categories"],
            strict=True,
        )
    )
    # under annotations
    assert all(
        annotations_out["category_id"] == annotations_in["category_id"] - 1
        for annotations_in, annotations_out in zip(
            input_dict["annotations"],
            output_dict["annotations"],
            strict=True,
        )
    )
