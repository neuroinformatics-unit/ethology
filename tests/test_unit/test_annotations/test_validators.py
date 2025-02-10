import json
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import jsonschema
import pytest

from ethology.annotations.json_schemas.utils import (
    _check_required_keys_in_dict,
    _check_required_properties_keys,
    _extract_properties_keys,
)
from ethology.annotations.validators import ValidCOCO, ValidVIA


@pytest.fixture()
def json_file_decode_error(tmp_path: Path) -> Path:
    """Return path to a JSON file with a decoding error."""
    json_file = tmp_path / "JSON_decode_error.json"
    with open(json_file, "w") as f:
        f.write("just-a-string")
    return json_file


@pytest.fixture()
def json_file_not_found_error(tmp_path: Path) -> Path:
    """Return path to a JSON file that does not exist."""
    return tmp_path / "JSON_file_not_found.json"


@pytest.fixture()
def VIA_file_schema_mismatch(
    annotations_test_data: dict,
    tmp_path: Path,
) -> Path:
    """Return path to a VIA JSON file that does not match its schema.

    Specifically, we modify the type of the "width" of the first bounding box
    in the first image, from "int" to "str"
    """
    # Read valid JSON file
    valid_VIA_file_sample_1 = annotations_test_data["VIA_JSON_sample_1.json"]
    with open(valid_VIA_file_sample_1) as f:
        data = json.load(f)

    # Modify file so that it doesn't match the corresponding schema
    # (make width a string)
    _, img_dict = list(data["_via_img_metadata"].items())[0]
    img_dict["regions"][0]["shape_attributes"]["width"] = "49"

    # Save the modified JSON to a new file
    out_json = tmp_path / f"{valid_VIA_file_sample_1.stem}_schema_error.json"
    with open(out_json, "w") as f:
        json.dump(data, f)
    return out_json


@pytest.fixture()
def COCO_file_schema_mismatch(
    annotations_test_data: dict,
    tmp_path: Path,
) -> Path:
    """Return path to a COCO JSON file that doesn't match its schema.

    Specifically, we modify the type of the object under the "annotations"
    key from "list of dicts" to "list"
    """
    # Read valid JSON file
    valid_COCO_file_sample_1 = annotations_test_data["COCO_JSON_sample_1.json"]
    with open(valid_COCO_file_sample_1) as f:
        data = json.load(f)

    # Modify file so that it doesn't match the corresponding schema
    data["annotations"] = [1, 2, 3]  # [d] for d in data["annotations"]]

    # save the modified json to a new file
    out_json = tmp_path / f"{valid_COCO_file_sample_1.stem}_schema_error.json"
    with open(out_json, "w") as f:
        json.dump(data, f)
    return out_json


@pytest.fixture()
def small_schema() -> dict:
    """Small schema with properties keys:
    ["a", "b", "b/b1", "c", "c/c1", "c/c2"].
    """
    return {
        "type": "object",
        "properties": {
            "a": {
                "type": "array",
                "items": {"type": "string"},
            },
            "b": {
                "type": "object",
                "properties": {"b1": {"type": "string"}},
            },
            "c": {
                "type": "object",
                "properties": {
                    "c1": {"type": "string"},
                    "c2": {"type": "string"},
                },
            },
        },
    }


@pytest.fixture()
def default_VIA_schema() -> dict:
    """Get default VIA schema."""
    from ethology.annotations.json_schemas.utils import _get_default_schema

    return _get_default_schema("VIA")


@pytest.fixture()
def default_COCO_schema() -> dict:
    """Get default COCO schema."""
    from ethology.annotations.json_schemas.utils import _get_default_schema

    return _get_default_schema("COCO")


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
            "'49' is not of type 'integer'",
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
    "schema, expected_properties_keys",
    [
        ("small_schema", ["a", "b", "b/b1", "c", "c/c1", "c/c2"]),
        (
            "default_VIA_schema",
            [
                "_via_attributes",
                "_via_attributes/file",
                "_via_attributes/region",
                "_via_attributes/region/default_options",
                "_via_attributes/region/description",
                "_via_attributes/region/options",
                "_via_attributes/region/type",
                "_via_data_format_version",
                "_via_image_id_list",
                "_via_img_metadata",
                "_via_img_metadata/file_attributes",
                "_via_img_metadata/filename",
                "_via_img_metadata/regions",
                "_via_img_metadata/regions/region_attributes",
                "_via_img_metadata/regions/shape_attributes",
                "_via_img_metadata/regions/shape_attributes/height",
                "_via_img_metadata/regions/shape_attributes/name",
                "_via_img_metadata/regions/shape_attributes/width",
                "_via_img_metadata/regions/shape_attributes/x",
                "_via_img_metadata/regions/shape_attributes/y",
                "_via_img_metadata/size",
                "_via_settings",
                "_via_settings/core",
                "_via_settings/project",
                "_via_settings/ui",
            ],
        ),
        (
            "default_COCO_schema",
            [
                "annotations",
                "annotations/area",
                "annotations/bbox",
                "annotations/category_id",
                "annotations/id",
                "annotations/image_id",
                "annotations/iscrowd",
                "categories",
                "categories/id",
                "categories/name",
                "categories/supercategory",
                "images",
                "images/file_name",
                "images/height",
                "images/id",
                "images/width",
                "info",
                "licenses",
            ],
        ),
    ],
)
def test_extract_properties_keys(
    schema: dict,
    expected_properties_keys: list,
    request: pytest.FixtureRequest,
):
    """Test the _extract_properties_keys helper function."""
    schema = request.getfixturevalue(schema)
    assert _extract_properties_keys(schema) == sorted(expected_properties_keys)


@pytest.mark.parametrize(
    (
        "list_required_keys, input_dict, additional_message, "
        "expected_exception, expected_message"
    ),
    [
        (
            ["images", "annotations", "categories"],
            {
                "images": [1, 2, 3],
                "annotations": [1, 2, 3],
                "categories": [1, 2, 3],
            },
            "",
            does_not_raise(),
            "",
        ),  # zero missing keys, and all keys map to non-empty values
        (
            ["images", "annotations", "categories"],
            {
                "images": [],
                "annotations": [1, 2, 3],
                "categories": [1, 2, 3],
            },
            "",
            pytest.raises(ValueError),
            "Empty value(s) found for the required key(s) ['images'].",
        ),  # zero missing keys, but one ("images") maps to empty values
        (
            ["images", "annotations", "categories"],
            {
                "images": [],
                "annotations": {},
                "categories": [1, 2, 3],
            },
            "",
            pytest.raises(ValueError),
            (
                "Empty value(s) found for the required key(s) "
                "['annotations', 'images']."
            ),
        ),  # zero missing keys, but two keys map to empty values
        (
            ["images", "annotations", "categories"],
            {"annotations": "", "categories": ""},
            "",
            pytest.raises(ValueError),
            "Required key(s) ['images'] not found.",
        ),  # one missing key
        (
            ["images", "annotations", "categories"],
            {"annotations": ""},
            "",
            pytest.raises(ValueError),
            "Required key(s) ['categories', 'images'] not found.",
        ),  # two missing keys
        (
            ["images", "annotations", "categories"],
            {"annotations": "", "categories": ""},
            "FOO",
            pytest.raises(ValueError),
            "Required key(s) ['images'] not foundFOO.",
        ),  # one missing key with additional message for missing keys
    ],
)
def test_check_required_keys_in_dict(
    list_required_keys: list,
    input_dict: dict,
    additional_message: str,
    expected_exception: pytest.raises,
    expected_message: str,
):
    """Test the _check_required_keys_in_dict helper function.

    The check verifies that the required keys are defined in the input
    dictionary and if they are, it checks that they do not map to empty
    values.
    """
    with expected_exception as excinfo:
        _check_required_keys_in_dict(
            list_required_keys, input_dict, additional_message
        )

    # Check error message
    if excinfo:
        assert expected_message in str(excinfo.value)


def test_check_required_properties_keys(small_schema: dict):
    """Test the _check_required_keys helper function."""
    # Define a sample schema from "small_schema"
    # with a "properties" key missing (e.g. "c/c2")
    small_schema["properties"]["c"]["properties"].pop("c2")

    # Define required "properties" keys
    required_keys = ["a", "b", "c/c2"]

    # Run check
    with pytest.raises(ValueError) as excinfo:
        _check_required_properties_keys(required_keys, small_schema)

    # Check error message
    assert "Required key(s) ['c/c2'] not found in schema" in str(excinfo.value)


@pytest.mark.parametrize(
    "input_file,",
    [
        "VIA_JSON_sample_1.json",
        "VIA_JSON_sample_2.json",
    ],
)
def test_required_keys_in_provided_VIA_schema(
    input_file: str, default_VIA_schema: dict, annotations_test_data: dict
):
    """Check the provided VIA schema contains the ValidVIA required keys."""
    # Get required keys from a VIA valid file
    filepath = annotations_test_data[input_file]
    valid_VIA = ValidVIA(path=filepath)
    required_VIA_keys = valid_VIA.required_keys

    # Map required keys to "properties" keys in schema
    map_required_to_properties_keys = {
        "main": "",
        "images": "_via_img_metadata",
        "regions": "_via_img_metadata/regions",
        "shape_attributes": "_via_img_metadata/regions/shape_attributes",
    }

    # Express required keys as required "properties" keys
    required_property_keys = [
        val if ky == "main" else f"{map_required_to_properties_keys[ky]}/{val}"
        for ky, values in required_VIA_keys.items()
        for val in values
    ]

    # Run check
    _check_required_properties_keys(
        required_property_keys,
        default_VIA_schema,
    )


@pytest.mark.parametrize(
    "input_file,",
    [
        "COCO_JSON_sample_1.json",
        "COCO_JSON_sample_2.json",
    ],
)
def test_required_keys_in_provided_COCO_schema(
    input_file: str, default_COCO_schema: dict, annotations_test_data: dict
):
    """Check the provided COCO schema contains the ValidCOCO required keys."""
    # Get required keys from a COCO valid file
    filepath = annotations_test_data[input_file]
    valid_COCO = ValidCOCO(path=filepath)
    required_COCO_keys = valid_COCO.required_keys

    # Prepare list of required "properties" keys with full paths
    required_properties_keys = [
        f"{level}/{ky}" if level != "main" else ky
        for level, required_keys in required_COCO_keys.items()
        for ky in required_keys
    ]

    # Run check
    _check_required_properties_keys(
        required_properties_keys,
        default_COCO_schema,
    )


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
    validator, input_file, expected_exception, annotations_test_data
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


def test_null_category_ID_behaviour(annotations_test_data):
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
    # Get path to test file
    filepath = annotations_test_data[
        "small_bboxes_non_unique_img_id_COCO.json"
    ]

    # Throws a schema validation error because image IDs are not unique
    with pytest.raises(ValueError) as excinfo:
        _ = ValidCOCO(path=filepath)

    assert str(excinfo.value) == (
        "The image IDs in the input COCO file are not unique. "
        "There are 4 image entries, but only 3 unique image IDs."
    )
