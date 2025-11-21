from contextlib import nullcontext as does_not_raise

import jsonschema
import numpy as np
import pytest
import xarray as xr

from ethology.validators.annotations import (
    ValidBboxAnnotationsDataset,
    ValidCOCO,
    ValidVIA,
)


@pytest.mark.parametrize(
    "input_file, validator",
    [
        ("VIA_JSON_sample_1.json", ValidVIA),
        ("VIA_JSON_sample_2.json", ValidVIA),
        ("COCO_JSON_sample_1.json", ValidCOCO),
        ("COCO_JSON_sample_2.json", ValidCOCO),
    ],
)
def test_validators_valid_input_files(
    input_file: str,
    validator: type[ValidVIA | ValidCOCO],
    annotations_test_data: dict,
):
    """Test the file validator with valid inputs."""
    filepath = annotations_test_data[input_file]
    with does_not_raise():
        validator(path=filepath)


@pytest.mark.parametrize(
    "invalid_input_file, validator, expected_exception, error_message",
    [
        (
            "json_file_decode_error",
            ValidVIA,
            pytest.raises(ValueError),
            "Error decoding JSON data from file",
        ),
        (
            "json_file_not_found_error",
            ValidVIA,
            pytest.raises(FileNotFoundError),
            "No such file or directory: ",
        ),
        (
            "json_file_decode_error",
            ValidCOCO,
            pytest.raises(ValueError),
            "Error decoding JSON data from file",
        ),
        (
            "json_file_not_found_error",
            ValidCOCO,
            pytest.raises(FileNotFoundError),
            "No such file or directory: ",
        ),
        (
            "VIA_file_schema_mismatch",
            ValidVIA,
            pytest.raises(jsonschema.exceptions.ValidationError),
            "'49' is not of type 'number'",
        ),
        (
            "COCO_file_schema_mismatch",
            ValidCOCO,
            pytest.raises(jsonschema.exceptions.ValidationError),
            "3 is not of type 'object'",
        ),
    ],
)
def test_validators_invalid_input_files(
    invalid_input_file: str,
    validator: type[ValidVIA | ValidCOCO],
    expected_exception: pytest.raises,
    error_message: str,
    request: pytest.FixtureRequest,
):
    """Test the validators throw the expected errors when passed invalid
    inputs.
    """
    invalid_json_file = request.getfixturevalue(invalid_input_file)

    with expected_exception as excinfo:
        validator(path=invalid_json_file)

    # Check that the error message contains expected string
    assert error_message in str(excinfo.value)

    # Check the error message contains file path
    if not isinstance(excinfo.value, jsonschema.exceptions.ValidationError):
        assert invalid_json_file.name in str(excinfo.value)


@pytest.mark.parametrize(
    "validator, input_file, expected_exception",
    [
        (
            ValidCOCO,
            "small_bboxes_no_cat_COCO.json",
            pytest.raises(ValueError),
        ),
        (ValidVIA, "small_bboxes_no_cat_VIA.json", does_not_raise()),
    ],
)
def test_no_categories_behaviour(
    validator: type[ValidVIA | ValidCOCO],
    input_file: str,
    expected_exception: pytest.raises,
    annotations_test_data: dict,
):
    """Test the behaviour of the validators when the input file does not
    specify any categories.

    If annotations are exported as a COCO file in the VIA tool, and no
    categories have been defined the required keys 'annotations' and
    'categories' map to empty values.

    If annotations are exported as a VIA file in the VIA tool, and no
    categories have been defined, there should be no error
    """
    filepath = annotations_test_data[input_file]

    with expected_exception as excinfo:
        _ = validator(path=filepath)

    if excinfo:
        assert (
            "Empty value(s) found for the required key(s) "
            "['annotations', 'categories']"
        ) in str(excinfo.value)


@pytest.mark.parametrize(
    "validator, input_file, expected_exception",
    [
        (
            ValidCOCO,
            "small_bboxes_no_supercat_COCO.json",
            does_not_raise(),
        ),
        (
            ValidVIA,
            "small_bboxes_no_supercat_VIA.json",
            does_not_raise(),
        ),
    ],
)
def test_no_supercategories_behaviour(
    validator: type[ValidVIA | ValidCOCO],
    input_file: str,
    expected_exception: pytest.raises,
    annotations_test_data: dict,
):
    """Test the behaviour of the validators when the input file does not
    specify any supercategories.

    COCO and VIA files exported with the VIA tool will always have a
    supercategory, but this can be set to " " (i.e., whitespace).

    COCO files not exported with the VIA tool may not have a supercategory.

    In this test we use a COCO file that does not have a supercategory, and
    a VIA file that has supercategory set to " " (i.e., whitespace).

    """
    filepath = annotations_test_data[input_file]

    with expected_exception as excinfo:
        _ = validator(path=filepath)

    if excinfo:
        assert (
            "Empty value(s) found for the required key(s) "
            "['annotations', 'categories']"
        ) in str(excinfo.value)


def test_null_category_ID_behaviour(annotations_test_data: dict):
    """Test the behaviour of the validators when the input file contains
    annotations with null category IDs.
    """
    # Get path to test file
    filepath = annotations_test_data["small_bboxes_null_catID_COCO.json"]

    # Throws a schema validation error because category IDs are not integer
    with pytest.raises(jsonschema.exceptions.ValidationError):
        _ = ValidCOCO(path=filepath)


def test_COCO_non_unique_image_IDs(annotations_test_data: dict):
    """Check the COCO validator throws an error when the input file contains
    non-unique image IDs.
    """
    filepath = annotations_test_data[
        "small_bboxes_non_unique_img_id_COCO.json"
    ]

    with pytest.raises(ValueError) as excinfo:
        _ = ValidCOCO(path=filepath)

    assert str(excinfo.value) == (
        "The image IDs in the input COCO file are not unique. "
        "There are 4 image entries, but only 3 unique image IDs."
    )


@pytest.mark.parametrize(
    "sample_dataset, expected_exception, expected_error_message",
    [
        (
            "valid_bbox_annotations_dataset",
            does_not_raise(),
            "",
        ),
        (
            "valid_bbox_annotations_dataset_extra_vars_and_dims",
            does_not_raise(),
            "",
        ),
        (
            xr.Dataset(
                coords={
                    "image_id": np.arange(3),
                    "space": np.arange(2),
                    "id": np.arange(2),
                },
                data_vars={
                    "position": (
                        ["image_id", "space", "id"],
                        np.zeros((3, 2, 2)),
                    ),
                    "shape": (
                        ["image_id", "space", "id", "foo"],
                        np.zeros((3, 2, 2, 1)),
                    ),
                },
            ),
            does_not_raise(),
            "",
        ),
        (
            {"position": [1, 2, 3], "shape": [4, 5, 6]},
            pytest.raises(TypeError),
            "Expected an xarray Dataset, but got <class 'dict'>.",
        ),
        (
            xr.Dataset(
                coords={
                    "image_id": np.arange(3),
                    "space": ["x", "y"],
                    "id": np.arange(2),
                },
                data_vars={
                    "position": (
                        ["image_id", "space", "id"],
                        np.zeros((3, 2, 2)),
                    ),
                },
            ),
            pytest.raises(ValueError),
            "Missing required data variables: ['shape']",
        ),
        (
            xr.Dataset(
                coords={
                    "image_id": np.arange(3),
                    "space": ["x", "y"],
                    "id": np.arange(2),
                },
                data_vars={
                    "foo": (
                        ["image_id", "space", "id"],
                        np.zeros((3, 2, 2)),
                    ),
                },
            ),
            pytest.raises(ValueError),
            "Missing required data variables: ['position', 'shape']",
        ),
        (
            xr.Dataset(
                coords={"image_id": np.arange(3), "id": np.arange(2)},
                data_vars={
                    "position": (["image_id", "id"], np.zeros((3, 2))),
                    "shape": (["image_id", "id"], np.zeros((3, 2))),
                },
            ),
            pytest.raises(ValueError),
            "Missing required dimensions: ['space']",
        ),
        (
            xr.Dataset(
                coords={
                    "foo": np.arange(3),
                    "bar": ["x", "y"],
                    "id": np.arange(2),
                },
                data_vars={
                    "position": (
                        ["foo", "bar", "id"],
                        np.zeros((3, 2, 2)),
                    ),
                    "shape": (
                        ["foo", "bar", "id"],
                        np.zeros((3, 2, 2)),
                    ),
                },
            ),
            pytest.raises(ValueError),
            "Missing required dimensions: ['image_id', 'space']",
        ),
        (
            xr.Dataset(
                coords={
                    "image_id": np.arange(3),
                    "space": np.arange(2),
                    "id": np.arange(2),
                },
                data_vars={
                    "position": (
                        ["image_id", "space", "id"],
                        np.zeros((3, 2, 2)),
                    ),
                    "shape": (
                        ["image_id", "id"],
                        np.zeros((3, 2)),
                    ),
                },
            ),
            pytest.raises(ValueError),
            (
                "Some data variables are missing required dimensions:"
                "\n  - data variable 'shape' is missing dimensions ['space']"
            ),
        ),
    ],
    ids=[
        "valid_bbox_annotations",
        "valid_bbox_annotations_extra_vars_and_dims",
        "valid_bbox_detections_extra_dims_in_shape_var",
        "invalid_bbox_annotations_type",
        "invalid_bbox_annotations_missing_data_var",
        "invalid_bbox_annotations_missing_multiple_data_vars",
        "invalid_bbox_annotations_missing_dimension",
        "invalid_bbox_annotations_missing_multiple_dimensions",
        "invalid_bbox_annotations_missing_dimension_in_data_var",
    ],
)
def test_validator_bbox_annotations_dataset(
    sample_dataset: str | dict,
    expected_exception: pytest.raises,
    expected_error_message: str,
    request: pytest.FixtureRequest,
):
    """Test bbox annotations dataset validation in various input scenarios."""
    # Get dataset to validate
    if isinstance(sample_dataset, str):
        dataset = request.getfixturevalue(sample_dataset)
    else:
        dataset = sample_dataset

    # Run validation and check exception
    with expected_exception as excinfo:
        validator = ValidBboxAnnotationsDataset(dataset=dataset)

    if excinfo:
        error_msg = str(excinfo.value)
        assert error_msg in expected_error_message
    else:
        assert validator.dataset is dataset
        assert validator.required_dims == {"image_id", "space", "id"}
        assert validator.required_data_vars == {
            "position": {"id", "image_id", "space"},
            "shape": {"id", "image_id", "space"},
        }
