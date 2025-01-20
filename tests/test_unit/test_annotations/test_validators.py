import json
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import jsonschema
import pytest

from ethology.annotations.validators import (
    ValidJSON,
    get_default_COCO_schema,
    get_default_VIA_schema,
)


# Schema fixtures
@pytest.fixture()
def invalid_VIA_schema() -> dict:
    """Return an invalid VIA schema."""
    invalid_VIA_schema = get_default_VIA_schema().copy()
    invalid_VIA_schema["type"] = "FOO"  # unsupported type
    return invalid_VIA_schema


@pytest.fixture()
def invalid_COCO_schema() -> dict:
    """Return an invalid COCO schema."""
    invalid_COCO_schema = get_default_COCO_schema().copy()
    invalid_COCO_schema["properties"]["images"]["type"] = (
        123  # should be a str
    )
    return invalid_COCO_schema


# Data file fixtures
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
    """Return path to a VIA JSON file that does not match its default schema.

    Specifically, we modify the type of the value of "width" in the first
    bounding box of the first image, from "int" to "str".
    """
    # Read valid VIA file
    valid_via_file_sample_1 = annotations_test_data["VIA_JSON_sample_1.json"]
    with open(valid_via_file_sample_1) as f:
        data = json.load(f)

    # Modify file so that it doesn't match its default schema
    _, img_dict = list(data["_via_img_metadata"].items())[0]
    img_dict["regions"][0]["shape_attributes"]["width"] = "49"

    # Save the modified VIA dictionary as a new file
    out_json = tmp_path / f"{valid_via_file_sample_1.stem}_schema_error.json"
    with open(out_json, "w") as f:
        json.dump(data, f)
    return out_json


@pytest.fixture()
def COCO_file_schema_mismatch(
    annotations_test_data: dict,
    tmp_path: Path,
) -> Path:
    """Return path to a COCO JSON file that doesn't match its default schema.

    Specifically, we modify the the object under the "annotations" key,
    to hold a list of integered, rather than the expected list of dictionaries.
    """
    # Read valid COCO file
    valid_coco_file_sample_1 = annotations_test_data["COCO_JSON_sample_1.json"]
    with open(valid_coco_file_sample_1) as f:
        data = json.load(f)

    # Modify file so that it doesn't match its default schema
    data["annotations"] = [1, 2, 3]

    # Save the modified COCO dictionary as a new file
    out_json = tmp_path / f"{valid_coco_file_sample_1.stem}_schema_error.json"
    with open(out_json, "w") as f:
        json.dump(data, f)
    return out_json


@pytest.mark.parametrize(
    "input_file, input_schema",
    [
        ("VIA_JSON_sample_1.json", None),
        ("VIA_JSON_sample_1.json", get_default_VIA_schema()),
        ("COCO_JSON_sample_1.json", None),
        ("COCO_JSON_sample_1.json", get_default_COCO_schema()),
        ("VIA_JSON_sample_2.json", None),
        ("VIA_JSON_sample_2.json", get_default_VIA_schema()),
        ("COCO_JSON_sample_2.json", None),
        ("COCO_JSON_sample_2.json", get_default_COCO_schema()),
    ],
)
def test_valid_json(
    input_file: str,
    input_schema: dict | None,
    annotations_test_data: dict,
):
    """Test the ValidJSON validator with valid inputs."""
    filepath = annotations_test_data[input_file]

    with does_not_raise():
        ValidJSON(
            path=filepath,
            schema=input_schema,
        )


@pytest.mark.parametrize(
    "input_file, input_schema, expected_exception, log_message",
    [
        (
            "json_file_decode_error",
            None,  # passing no schema
            pytest.raises(ValueError),
            "Error decoding JSON data from file",  # decoding error
        ),
        (
            "json_file_decode_error",
            get_default_VIA_schema(),  # passing a schema schema
            pytest.raises(ValueError),
            "Error decoding JSON data from file",  # decoding error
        ),
        (
            "json_file_not_found_error",
            None,  # passing no schema
            pytest.raises(FileNotFoundError),
            "File not found",  # file error
        ),
        (
            "json_file_not_found_error",
            get_default_VIA_schema(),  # passing a schema
            pytest.raises(FileNotFoundError),
            "File not found",  # file error
        ),
        (
            "VIA_file_schema_mismatch",
            get_default_VIA_schema(),
            pytest.raises(jsonschema.exceptions.ValidationError),
            "'49' is not of type 'integer'\n\n",  # file does not match schema
        ),
        (
            "COCO_file_schema_mismatch",
            get_default_COCO_schema(),
            pytest.raises(jsonschema.exceptions.ValidationError),
            "3 is not of type 'object'\n\n",  # file does not match schema
        ),
    ],
)
def test_valid_json_invalid_inputs(
    input_file: str,
    input_schema: dict | None,
    expected_exception: pytest.raises,
    log_message: str,
    request: pytest.FixtureRequest,
):
    """Test the ValidJSON validator throws the expected exceptions when passed
    invalid inputs.

    The invalid inputs cases covered in this test are:
    - a JSON file that cannot be decoded
    - a JSON file that does not exist
    - a JSON file that does not match the provided (correct) schema
    """
    # Read invalid input JSON file
    invalid_json_file = request.getfixturevalue(input_file)

    # Run validation
    with expected_exception as excinfo:
        ValidJSON(
            path=invalid_json_file,
            schema=input_schema,
        )

    # Check that the error message contains expected string
    assert log_message in str(excinfo.value)

    # If the error is not a schema-file mismatch, additionally check the
    # error message contains the file path
    if not isinstance(excinfo.value, jsonschema.exceptions.ValidationError):
        assert invalid_json_file.name in str(excinfo.value)


@pytest.mark.parametrize(
    "input_file, invalid_schema",
    [
        ("VIA_JSON_sample_1.json", "invalid_VIA_schema"),
        ("COCO_JSON_sample_1.json", "invalid_COCO_schema"),
    ],
)
def test_valid_json_invalid_schema(
    input_file: str,
    invalid_schema: str,
    annotations_test_data: dict,
    request: pytest.FixtureRequest,
):
    """Test the ValidJSON validator throws an error when the schema is
    invalid.
    """
    # Read input data
    input_file_path = annotations_test_data[input_file]
    invalid_schema_dict = request.getfixturevalue(invalid_schema)

    # Run validation
    with pytest.raises(jsonschema.exceptions.SchemaError) as excinfo:
        ValidJSON(
            path=input_file_path,
            schema=invalid_schema_dict,
        )

    # Check the error message is as expected
    assert "is not valid under any of the given schemas" in str(excinfo.value)
