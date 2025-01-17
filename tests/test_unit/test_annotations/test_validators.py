import json
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import jsonschema
import pytest

from ethology.annotations.validators import (
    ValidCOCO,
    ValidJSON,
    ValidVIA,
    get_default_coco_schema,
    get_default_via_schema,
)


# Schema fixtures
@pytest.fixture()
def invalid_via_schema() -> dict:
    invalid_VIA_schema = get_default_via_schema().copy()
    invalid_VIA_schema["type"] = "FOO"
    return invalid_VIA_schema


@pytest.fixture()
def invalid_coco_schema() -> dict:
    invalid_COCO_schema = get_default_coco_schema().copy()
    invalid_COCO_schema["properties"]["images"]["type"] = 123
    return invalid_COCO_schema


# Files fixtures
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
def via_file_schema_mismatch(
    annotations_test_data: dict,
    tmp_path: Path,
) -> Path:
    """Return path to a VIA JSON file that does not match its schema.

    Specifically, we modify the type of the "width" of the first bounding box
    in the first image, from "int" to "str"
    """
    # Read valid JSON file
    valid_via_file_sample_1 = annotations_test_data["VIA_JSON_sample_1.json"]
    with open(valid_via_file_sample_1) as f:
        data = json.load(f)

    # Modify file so that it doesn't match the corresponding schema
    # (make width a string)
    _, img_dict = list(data["_via_img_metadata"].items())[0]
    img_dict["regions"][0]["shape_attributes"]["width"] = "49"

    # Save the modified JSON to a new file
    out_json = tmp_path / f"{valid_via_file_sample_1.stem}_schema_error.json"
    with open(out_json, "w") as f:
        json.dump(data, f)
    return out_json


@pytest.fixture()
def coco_file_schema_mismatch(
    annotations_test_data: dict,
    tmp_path: Path,
) -> Path:
    """Return path to a COCO JSON file that doesn't match its schema.

    Specifically, we modify the type of the object under the "annotations"
    key from "list of dicts" to "list"
    """
    # Read valid JSON file
    valid_coco_file_sample_1 = annotations_test_data["COCO_JSON_sample_1.json"]
    with open(valid_coco_file_sample_1) as f:
        data = json.load(f)

    # Modify file so that it doesn't match the corresponding schema
    data["annotations"] = [1, 2, 3]  # [d] for d in data["annotations"]]

    # save the modified json to a new file
    out_json = tmp_path / f"{valid_coco_file_sample_1.stem}_schema_error.json"
    with open(out_json, "w") as f:
        json.dump(data, f)
    return out_json


@pytest.mark.parametrize(
    "input_file, input_schema",
    [
        ("VIA_JSON_sample_1.json", None),
        ("VIA_JSON_sample_1.json", get_default_via_schema()),
        ("COCO_JSON_sample_1.json", None),
        ("COCO_JSON_sample_1.json", get_default_coco_schema()),
        ("VIA_JSON_sample_2.json", None),
        ("VIA_JSON_sample_2.json", get_default_via_schema()),
        ("COCO_JSON_sample_2.json", None),
        ("COCO_JSON_sample_2.json", get_default_coco_schema()),
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
    "invalid_input_file, input_schema, expected_exception, log_message",
    [
        (
            "json_file_decode_error",
            None,  # no schema
            pytest.raises(ValueError),
            "Error decoding JSON data from file",  # decoding error
        ),
        (
            "json_file_decode_error",
            get_default_via_schema(),  # with schema
            pytest.raises(ValueError),
            "Error decoding JSON data from file",  # decoding error with schema
        ),
        (
            "json_file_not_found_error",
            None,  # no schema
            pytest.raises(FileNotFoundError),
            "File not found",  # file error
        ),
        (
            "json_file_not_found_error",
            get_default_via_schema(),  # with schema
            pytest.raises(FileNotFoundError),
            "File not found",  # file error with schema
        ),
        (
            "via_file_schema_mismatch",
            get_default_via_schema(),
            pytest.raises(jsonschema.exceptions.ValidationError),
            "'49' is not of type 'integer'\n\n",  # file does not match schema
        ),
        (
            "coco_file_schema_mismatch",
            get_default_coco_schema(),
            pytest.raises(jsonschema.exceptions.ValidationError),
            "3 is not of type 'object'\n\n",  # file does not match schema
        ),
    ],
)
def test_valid_json_invalid_files(
    invalid_input_file: str,
    input_schema: dict | None,
    expected_exception: pytest.raises,
    log_message: str,
    request: pytest.FixtureRequest,
):
    """Test the ValidJSON validator throws the expected errors when passed
    invalid inputs.

    The invalid inputs cases covered in this test are:
    - a JSON file that cannot be decoded
    - a JSON file that does not exist
    - a JSON file that does not match the given (correct) schema
    """
    invalid_json_file = request.getfixturevalue(invalid_input_file)

    with expected_exception as excinfo:
        ValidJSON(
            path=invalid_json_file,
            schema=input_schema,
        )

    # Check that the error message contains expected string
    assert log_message in str(excinfo.value)

    # If error is not a schema-mismatch, check the error message contains
    # file path
    if not isinstance(excinfo.value, jsonschema.exceptions.ValidationError):
        assert invalid_json_file.name in str(excinfo.value)


@pytest.mark.parametrize(
    "input_file, invalid_schema",
    [
        ("VIA_JSON_sample_1.json", "invalid_via_schema"),
        ("COCO_JSON_sample_1.json", "invalid_coco_schema"),
    ],
)
def test_valid_json_invalid_schema(
    input_file, invalid_schema, annotations_test_data, request
):
    """Test the ValidJSON validator throws an error when the schema is
    invalid.
    """
    input_file = annotations_test_data[input_file]
    invalid_schema = request.getfixturevalue(invalid_schema)

    with pytest.raises(jsonschema.exceptions.SchemaError) as excinfo:
        ValidJSON(
            path=input_file,
            schema=invalid_schema,
        )

    # Check the error message is as expected
    assert "is not valid under any of the given schemas" in str(excinfo.value)


@pytest.mark.parametrize(
    "input_file,",
    [
        "VIA_JSON_sample_1.json",
        "VIA_JSON_sample_2.json",
    ],
)
def test_valid_via(input_file: str, annotations_test_data: dict):
    """Test the VIA validator with valid inputs."""
    filepath = annotations_test_data[input_file]
    with does_not_raise():
        ValidVIA(path=filepath)


@pytest.mark.parametrize(
    "invalid_input_file, expected_exception, log_message",
    [
        (
            "json_file_decode_error",
            pytest.raises(ValueError),
            "Error decoding JSON data from file",
        ),
        (
            "json_file_not_found_error",
            pytest.raises(FileNotFoundError),
            "File not found",
        ),
        (
            "via_file_schema_mismatch",
            pytest.raises(jsonschema.exceptions.ValidationError),
            "'49' is not of type 'integer'",
        ),
    ],
)
def test_valid_via_invalid_files(
    invalid_input_file: str,
    expected_exception: pytest.raises,
    log_message: str,
    request: pytest.FixtureRequest,
):
    """Test the VIA validator throwS the expected errors when passed invalid
    inputs.
    """
    invalid_json_file = request.getfixturevalue(invalid_input_file)

    with expected_exception as excinfo:
        ValidVIA(path=invalid_json_file)

    # Check that the error message contains expected string
    assert log_message in str(excinfo.value)

    # Check the error message contains file path
    # assert invalid_json_file.name in str(excinfo.value)
    if not isinstance(excinfo.value, jsonschema.exceptions.ValidationError):
        assert invalid_json_file.name in str(excinfo.value)


@pytest.mark.parametrize(
    "input_file",
    [
        "COCO_JSON_sample_1.json",
        "COCO_JSON_sample_2.json",
    ],
)
def test_valid_coco(input_file: str, annotations_test_data: dict):
    """Test the COCO validator with valid inputs."""
    filepath = annotations_test_data[input_file]
    with does_not_raise():
        # run the validator
        ValidCOCO(path=filepath)
