import json
from contextlib import nullcontext as does_not_raise

import pytest

from ethology.annotations.json_schemas import (
    COCO_UNTRACKED_SCHEMA,
    VIA_UNTRACKED_SCHEMA,
)
from ethology.annotations.validators import ValidJSON


@pytest.fixture()
def json_file_with_decode_error(tmp_path):
    """Return factory of paths to JSON files with a decoding error."""
    json_file = tmp_path / "JSON_decode_error.json"
    with open(json_file, "w") as f:
        f.write("just-a-string")
    return json_file


@pytest.fixture()
def json_file_with_not_found_error(tmp_path):
    """Return the path to a JSON file that does not exist."""
    return tmp_path / "JSON_file_not_found.json"


@pytest.fixture()
def via_json_file_with_schema_error(tmp_path, annotations_test_data):
    """Return path to a JSON file that doesn't match the expected schema."""
    # read valid json file
    via_json_valid_filepath = annotations_test_data["VIA_JSON_sample_1.json"]
    with open(via_json_valid_filepath) as f:
        data = json.load(f)

    # change type of specific keys
    # - change "_via_image_id_list" from list of strings to list of integers
    # TODO: what if I change several?
    data["_via_image_id_list"] = list(range(len(data["_via_image_id_list"])))

    # save the modified data to a new file under tmp_path
    out_json = tmp_path / "VIA_JSON_schema_error.json"
    with open(out_json, "w") as f:
        json.dump(data, f)
    return out_json


@pytest.fixture()
def coco_json_file_with_schema_error(
    tmp_path,
    annotations_test_data,
):
    """Return path to a JSON file that doesn't match the expected schema."""
    # read valid json file
    via_json_valid_filepath = annotations_test_data["COCO_JSON_sample_1.json"]
    with open(via_json_valid_filepath) as f:
        data = json.load(f)

    # change "annotations" from list of dicts to list of lists
    # TODO: what if I change several?
    data["annotations"] = [[d] for d in data["annotations"]]

    # save the modified data to a new file under tmp_path
    out_json = tmp_path / "VIA_JSON_schema_error.json"
    with open(out_json, "w") as f:
        json.dump(data, f)
    return out_json


@pytest.mark.parametrize(
    "input_json_file_suffix",
    ["1", "2"],
)
@pytest.mark.parametrize(
    "input_file_standard, input_schema",
    [
        ("VIA", VIA_UNTRACKED_SCHEMA),
        ("VIA", None),
        ("COCO", COCO_UNTRACKED_SCHEMA),
        ("COCO", None),
    ],
)
def test_valid_json(
    annotations_test_data,
    input_file_standard,
    input_json_file_suffix,
    input_schema,
):
    """Test the ValidJSON validator with valid files."""
    input_json_file = (
        f"{input_file_standard}_JSON_sample_{input_json_file_suffix}.json"
    )
    input_json_file = annotations_test_data[input_json_file]
    with does_not_raise():
        ValidJSON(path=input_json_file, schema=input_schema)


@pytest.mark.parametrize(
    "invalid_json_file_str, input_schema, expected_exception, log_message",
    [
        (
            "json_file_with_decode_error",
            VIA_UNTRACKED_SCHEMA,  # should be independent of schema
            pytest.raises(ValueError),
            "Error decoding JSON data from file: {}.",
        ),
        (
            "json_file_with_not_found_error",
            VIA_UNTRACKED_SCHEMA,  # should be independent of schema
            pytest.raises(FileNotFoundError),
            "File not found: {}.",
        ),
        (
            "via_json_file_with_schema_error",
            VIA_UNTRACKED_SCHEMA,
            pytest.raises(ValueError),
            "The JSON data does not match the provided schema: {}.",
        ),
        (
            "coco_json_file_with_schema_error",
            COCO_UNTRACKED_SCHEMA,
            pytest.raises(ValueError),
            "The JSON data does not match the provided schema: {}.",
        ),
    ],
)
def test_valid_json_error(
    invalid_json_file_str,
    input_schema,
    expected_exception,
    log_message,
    request,
):
    """Test the ValidJSON validator throws the expected error."""
    invalid_json_file = request.getfixturevalue(invalid_json_file_str)

    with expected_exception as excinfo:
        ValidJSON(path=invalid_json_file, schema=input_schema)

    if "schema" in invalid_json_file_str:
        assert str(excinfo.value) == log_message.format(input_schema)
    else:
        assert str(excinfo.value) == log_message.format(invalid_json_file)


# @pytest.mark.parametrize(
#     "valid_json_file, input_schema",
#     [
#         ("VIA_JSON_sample_1.json", VIA_UNTRACKED_SCHEMA),
#         ("COCO_JSON_sample_1.json", COCO_UNTRACKED_SCHEMA),
#     ],
# )
# @pytest.mark.parametrize(
#     "invalid_json_factory, expected_exception, log_message",
#     [
#         (
#             "get_json_file_with_schema_error",
#             pytest.raises(ValueError),
#             "The JSON data does not match the provided schema: {}.",
#         ),
#     ],
# )
# def test_valid_json_schema_error(
#     valid_json_file,
#     input_schema,
#     invalid_json_factory,
#     expected_exception,
#     log_message,
#     tmp_path,
#     request,
# ):
#     """Test the ValidJSON validator throws the expected error."""
#     invalid_json_factory = request.getfixturevalue(invalid_json_factory)
#     invalid_json_file = invalid_json_factory(valid_json_file)

#     with expected_exception as excinfo:
#         ValidJSON(path=invalid_json_file, schema=input_schema)

#     if log_message:
#         assert str(excinfo.value) == log_message.format(input_schema)
