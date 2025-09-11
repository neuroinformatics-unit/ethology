import copy
import json
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from pathlib import Path
from typing import Literal

import pandas as pd
import pandera.pandas as pa
import pytest

from ethology.io.annotations.load_bboxes import from_files
from ethology.io.annotations.save_bboxes import (
    _add_COCO_data_to_df,
    _create_COCO_dict,
    _get_raw_df_from_ds,
    to_COCO_file,
)
from ethology.io.annotations.validate import ValidBboxesDataFrameCOCO


def read_JSON_as_dict(file_path: str | Path) -> dict:
    """Read a JSON file and return its content as a dictionary."""
    with open(file_path) as file:
        return json.load(file)


def check_dict_in_list_of_dicts(
    input_dict: dict, list_dicts: list, keys_to_exclude: list | None = None
) -> bool:
    """Check if a dictionary is in a list of dictionaries, considering only
    certain keys in the comparison.
    """
    # Prepare list of keys for comparison
    # (we assume all dictionaries in the list have the same keys as the first)
    common_keys = set(input_dict.keys()).intersection(list_dicts[0].keys())
    keys_to_exclude = keys_to_exclude or []
    keys_to_compare = common_keys.difference(keys_to_exclude)

    # Define modified list removing keys to exclude and non-common keys
    list_dicts_modif = [{k: d[k] for k in keys_to_compare} for d in list_dicts]
    input_dict_modif = {k: input_dict[k] for k in keys_to_compare}

    # Check if input dictionary is in the list of dictionaries
    return input_dict_modif in list_dicts_modif


def assert_list_of_dicts_match(
    list_dicts_1: list, list_dicts_2: list, keys_to_exclude: list | None = None
):
    """Assert two lists of dictionaries are equal.

    We do this by checking that both lists are the same length, and
    each dictionary in list 1 exists in list 2.
    """
    # Check same length
    assert len(list_dicts_1) == len(list_dicts_2)

    # Check each dictionary in list 1 is in list 2
    assert all(
        check_dict_in_list_of_dicts(dict_1, list_dicts_2, keys_to_exclude)
        for dict_1 in list_dicts_1
    )


@pytest.fixture
def sample_bboxes_df() -> Callable:
    """Return a factory function for a COCO-exportable bboxes dataframe.

    The factory function can be called with the `columns_to_drop` parameter
    """

    def _sample_bboxes_df_drop(
        columns_to_drop: list | None = None,
    ) -> pd.DataFrame:
        """Return a COCO-exportable bboxes df with specified columns dropped.

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
                "x_min": {0: 963.0, 1: 376.0, 2: 458.0},
                "y_min": {0: 283.0, 1: 314.0, 2: 329.0},
                "width": {0: 302.0, 1: 301.0, 2: 301.0},
                "height": {0: 172.0, 1: 123.0, 2: 131.0},
                "area": {0: 51944.0, 1: 37023.0, 2: 39431.0},
                "supercategory": {0: "animal", 1: "animal", 2: "animal"},
                "category": {0: "crab", 1: "crab", 2: "crab"},
                "category_id": {0: 1, 1: 1, 2: 1},
                "iscrowd": {0: 0, 1: 0, 2: 0},
                "segmentation": {
                    0: [
                        [
                            963.0,
                            283.0,
                            1265.0,
                            283.0,
                            1265.0,
                            455.0,
                            963.0,
                            455.0,
                        ]
                    ],
                    1: [
                        [
                            376.0,
                            314.0,
                            677.0,
                            314.0,
                            677.0,
                            437.0,
                            376.0,
                            437.0,
                        ]
                    ],
                    2: [
                        [
                            458.0,
                            329.0,
                            759.0,
                            329.0,
                            759.0,
                            460.0,
                            458.0,
                            460.0,
                        ]
                    ],
                },
                "bbox": {
                    0: [963.0, 283.0, 302.0, 172.0],
                    1: [376.0, 314.0, 301.0, 123.0],
                    2: [458.0, 329.0, 301.0, 131.0],
                },
                "image_width": {0: 1280, 1: 1280, 2: 1280},
                "image_height": {0: 720, 1: 720, 2: 720},
            }
        ).set_index("annotation_id", drop=False)

        # Validate as COCO-exportable
        df = ValidBboxesDataFrameCOCO.validate(df)

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
            pytest.raises(pa.errors.BackendNotFoundError),
            "Backend not found for backend, class: "
            "(<class 'pandera.api.pandas.container.DataFrameSchema'>, "
            "<class 'list'>)",
        ),
        (
            pd.DataFrame(),
            pytest.raises(pa.errors.SchemaError),
            "column 'annotation_id' not in dataframe. "
            "Columns in dataframe: []",
        ),
        (
            pd.DataFrame(
                {
                    "annotation_id": [1, 2, 3],
                }
            ).set_index("annotation_id", drop=False),
            pytest.raises(pa.errors.SchemaError),
            "column 'image_id' not in dataframe. "
            "Columns in dataframe: ['annotation_id']",
        ),
        (
            pd.DataFrame(
                {
                    "annotation_id": [1, 2, 3],
                    "image_id": [0, 0, 0],
                    "image_filename": ["00000.jpg", "00000.jpg", "00000.jpg"],
                }
            ).set_index("annotation_id", drop=False),
            pytest.raises(pa.errors.SchemaError),
            "column 'image_width' not in dataframe. "
            "Columns in dataframe: "
            "['annotation_id', 'image_id', 'image_filename']",
        ),
        (
            "sample_bboxes_df",
            does_not_raise(),
            "",
        ),
    ],
)
def test_validate_bboxes_df_COCO(
    df: pd.DataFrame,
    expected_exception: pytest.raises,
    expected_error_message: str,
    request: pytest.FixtureRequest,
):
    """Test the validation of the COCO-exportable bboxes dataframe throws the
    expected errors.
    """
    if not expected_error_message:
        df_factory = request.getfixturevalue(df)
        df = df_factory()
    with expected_exception as excinfo:
        ValidBboxesDataFrameCOCO(df)
    if excinfo:
        assert expected_error_message in str(excinfo.value)


@pytest.mark.parametrize(
    "input_file",
    [
        "small_bboxes_COCO_subset.json",  # includes image shape data
        "small_bboxes_VIA_subset.json",  # includes image shape data
    ],
)
@pytest.mark.parametrize(
    "drop_variables",
    [
        True,
        False,
    ],
)
def test_get_raw_df_from_ds(
    annotations_test_data: dict, input_file: str, drop_variables: bool
):
    """Test the function that gets the raw dataframe derived from the xarray
    dataset fills in the appropriate category values, and includes the image
    shape columns if present in the original dataset.
    """
    input_file = annotations_test_data[input_file]
    format: Literal["VIA", "COCO"] = (
        "VIA" if "VIA" in str(input_file) else "COCO"
    )
    ds = from_files(input_file, format=format)

    # Drop data arrays if specified
    if drop_variables:
        vars_to_drop = [
            var
            for var in ["category", "image_shape"]
            if var in list(ds.data_vars.keys())
        ]
        ds = ds.drop_vars(vars_to_drop)  # type: ignore

    # Get raw dataframe
    df_raw = _get_raw_df_from_ds(ds)

    # The "category" column should always be present in the raw dataframe,
    # even if the category array was not present in the original dataset
    list_expected_columns = [
        "image_id",
        "id",
        "category",
        "position_x",
        "position_y",
        "shape_x",
        "shape_y",
    ]

    # The image_shape_x and image_shape_y columns should be present only if
    # the image shape array was present in the original dataset
    if "image_shape" in list(ds.data_vars.keys()):
        list_expected_columns.extend(["image_shape_x", "image_shape_y"])

    # Check columns
    assert sorted(df_raw.columns.tolist()) == sorted(list_expected_columns)


def test_add_COCO_data_to_df(annotations_test_data: dict):
    """Test the function that prepares a COCO-exportable bboxes dataframe
    fills in any required columns that may be missing.
    """
    # Read input file as bboxes dataset
    input_file = annotations_test_data["small_bboxes_COCO.json"]
    ds = from_files(input_file, format="COCO")

    # Get raw dataframe
    df_raw = _get_raw_df_from_ds(ds)

    # Fill in missing columns with defaults
    df_output = _add_COCO_data_to_df(df_raw, ds.attrs)

    # Check image columns
    assert all(
        df_output["image_filename"]
        == df_raw["image_id"].map(ds.attrs["map_image_id_to_filename"])
    )
    assert all(df_output["image_width"] == 1280)
    assert all(df_output["image_height"] == 720)

    # bbox
    assert all(
        df_output["bbox"]
        == df_raw.apply(
            lambda row: [
                row["position_x"] - row["shape_x"] / 2,  # xmin
                row["position_y"] - row["shape_y"] / 2,  # ymin
                row["shape_x"],  # width
                row["shape_y"],  # height
            ],
            axis=1,
        )
    )
    assert all(df_output["area"] == df_raw["shape_x"] * df_raw["shape_y"])
    assert all(
        df_output["segmentation"]
        == df_raw.apply(
            lambda row: [
                [
                    row["position_x"] - row["shape_x"] / 2,  # xmin
                    row["position_y"] - row["shape_y"] / 2,  # ymin, top-left
                    row["position_x"] + row["shape_x"] / 2,  # xmax
                    row["position_y"] - row["shape_y"] / 2,  # ymin, top-right
                    row["position_x"] + row["shape_x"] / 2,  # xmax
                    row["position_y"]
                    + row["shape_y"] / 2,  # ymax, bottom-right
                    row["position_x"] - row["shape_x"] / 2,  # xmin
                    row["position_y"]
                    + row["shape_y"] / 2,  # ymax, bottom-left
                ]
            ],
            axis=1,
        )
    )

    # category
    assert "category_id" in df_output.columns
    assert all(
        df_output["category"]
        == df_output["category_id"].map(ds.attrs["map_category_to_str"])
    )
    assert all(df_output["supercategory"] == "")

    # other
    assert all(df_output["iscrowd"] == 0)


def test_create_COCO_dict(sample_bboxes_df: Callable):
    """Test the function that transforms the modified bboxes dataframe to
    a COCO dictionary.
    """
    # Take an COCO-exportable df
    df = sample_bboxes_df()

    # Extract COCO dictionary
    COCO_dict = _create_COCO_dict(df)

    # Check type and sections
    assert isinstance(COCO_dict, dict)
    assert all(x in COCO_dict for x in ["images", "categories", "annotations"])

    # Check keys in each section
    map_df_columns_to_coco = copy.deepcopy(
        ValidBboxesDataFrameCOCO.map_df_columns_to_COCO_fields()
    )
    for section, section_mapping in map_df_columns_to_coco.items():
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
        # small COCO file, image shape data as non-zero integers
        "COCO_JSON_sample_1.json",
        # medium COCO file, no image shape data (width and height are 0)
        "small_bboxes_no_supercat_COCO.json",
        # small COCO file, no supercategory data
    ],
)
def test_to_COCO_file(
    filename: str, annotations_test_data: dict, tmp_path: Path
):
    """Test the function that exports a bboxes dataset to a COCO JSON file."""
    # Read input file as bboxes dataset
    input_file = annotations_test_data[filename]
    ds = from_files(input_file, format="COCO")

    # Export dataset to COCO format
    output_file = to_COCO_file(ds, output_filepath=tmp_path / "output.json")

    # Read input and output files as dictionaries
    input_dict = read_JSON_as_dict(input_file)
    output_dict = read_JSON_as_dict(output_file)

    # Format "area" and "bbox" as floats in input dict
    # (we export them as floats)
    input_dict["annotations"] = [
        {
            **ann,
            "area": float(ann["area"]),
            "bbox": [float(x) for x in ann["bbox"]],
        }
        for ann in input_dict["annotations"]
    ]

    # Check lists of "categories" dictionaries match
    # We exclude supercategory because we do not retain it in the dataset
    assert_list_of_dicts_match(
        input_dict["categories"],
        output_dict["categories"],
        keys_to_exclude=["supercategory"],
    )

    # Check lists of "images" dictionaries match
    # We exclude id because when exporting COCO file from VIA,
    # it tries to extract image ID from image filename.
    # This means it will be different from the image "id"
    # in the dataset.
    assert_list_of_dicts_match(
        input_dict["images"],
        output_dict["images"],
        keys_to_exclude=["height", "width", "id"],
    )

    # Check lists of "annotations" dictionaries match
    # We exclude the ids because we export a 0-based
    # annotation ID and image ID under "id" and "image_id"
    # respectively.
    assert_list_of_dicts_match(
        input_dict["annotations"],
        output_dict["annotations"],
        keys_to_exclude=["id", "image_id"],
    )
