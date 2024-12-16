import json
from contextlib import nullcontext as does_not_raise

import jsonschema
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


def _json_file_with_schema_error(out_parent_path, json_valid_path):
    """Return path to a JSON file that doesn't match the expected schema."""
    # read valid json file
    with open(json_valid_path) as f:
        data = json.load(f)

    # modify so that it doesn't match the corresponding schema
    if "VIA" in json_valid_path.name:
        # change "width" of a bounding box from int to float
        data["_via_img_metadata"][
            "09.08_09.08.2023-01-Left_frame_001764.png15086122"
        ]["regions"][0]["shape_attributes"]["width"] = 49.5
    elif "COCO" in json_valid_path.name:
        # change "annotations" from list of dicts to list of lists
        data["annotations"] = [[d] for d in data["annotations"]]

    # save the modified json to a new file
    out_json = out_parent_path / f"{json_valid_path.name}_schema_error.json"
    with open(out_json, "w") as f:
        json.dump(data, f)
    return out_json


@pytest.fixture()
def via_json_file_with_schema_error(
    tmp_path,
    annotations_test_data,
):
    """Return path to a VIA JSON file that doesn't match its schema."""
    return _json_file_with_schema_error(
        tmp_path,
        annotations_test_data["VIA_JSON_sample_1.json"],
    )


@pytest.fixture()
def coco_json_file_with_schema_error(
    tmp_path,
    annotations_test_data,
):
    """Return path to a COCO JSON file that doesn't match its schema."""
    return _json_file_with_schema_error(
        tmp_path,
        annotations_test_data["COCO_JSON_sample_1.json"],
    )


@pytest.mark.parametrize(
    "input_file_standard, input_schema",
    [
        ("VIA", None),
        ("VIA", VIA_UNTRACKED_SCHEMA),
        ("COCO", None),
        ("COCO", COCO_UNTRACKED_SCHEMA),
    ],
)
@pytest.mark.parametrize(
    "input_json_file_suffix",
    ["JSON_sample_1.json", "JSON_sample_2.json"],
)
def test_valid_json(
    input_file_standard,
    input_json_file_suffix,
    input_schema,
    annotations_test_data,
):
    """Test the ValidJSON validator with valid files."""
    # get path to file
    filepath = annotations_test_data[
        f"{input_file_standard}_{input_json_file_suffix}"
    ]

    with does_not_raise():
        ValidJSON(
            path=filepath,
            schema=input_schema,
        )


@pytest.mark.parametrize(
    "invalid_json_file_str, input_schema, expected_exception, log_message",
    [
        (
            "json_file_with_decode_error",
            None,  # should be independent of schema
            pytest.raises(ValueError),
            "Error decoding JSON data from file: {}.",
        ),
        (
            "json_file_with_not_found_error",
            None,  # should be independent of schema
            pytest.raises(FileNotFoundError),
            "File not found: {}.",
        ),
        (
            "via_json_file_with_schema_error",
            VIA_UNTRACKED_SCHEMA,
            pytest.raises(jsonschema.exceptions.ValidationError),
            "49.5 is not of type 'integer'\n\n"
            "Failed validating 'type' in "
            "schema['properties']['_via_img_metadata']['additionalProperties']"
            "['properties']['regions']['items']['properties']"
            "['shape_attributes']['properties']['width']:\n"
            "    {'type': 'integer'}\n\n"
            "On instance['_via_img_metadata']"
            "['09.08_09.08.2023-01-Left_frame_001764.png15086122']['regions']"
            "[0]['shape_attributes']['width']:\n"
            "    49.5",
        ),
        (
            "coco_json_file_with_schema_error",
            COCO_UNTRACKED_SCHEMA,
            pytest.raises(jsonschema.exceptions.ValidationError),
            "[{'area': 432, 'bbox': [1278, 556, 16, 27], 'category_id': 1, "
            "'id': 8917, 'image_id': 199, 'iscrowd': 0}] is not of type "
            "'object'\n\n"
            "Failed validating 'type' in "
            "schema['properties']['annotations']['items']:\n"
            "    {'type': 'object',\n"
            "     'properties': {'id': {'type': 'integer'},\n"
            "                    'image_id': {'type': 'integer'},\n"
            "                    'bbox': {'type': 'array', 'items': "
            "{'type': 'integer'}},\n"
            "                    'category_id': {'type': 'integer'},\n"
            "                    'area': {'type': 'integer'},\n"
            "                    'iscrowd': {'type': 'integer'}}}\n\n"
            "On instance['annotations'][4343]:\n"
            "    [{'area': 432,\n"
            "      'bbox': [1278, 556, 16, 27],\n"
            "      'category_id': 1,\n"
            "      'id': 8917,\n"
            "      'image_id': 199,\n"
            "      'iscrowd': 0}]",
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
    """Test the ValidJSON validator throws the expected errors."""
    invalid_json_file = request.getfixturevalue(invalid_json_file_str)

    with expected_exception as excinfo:
        ValidJSON(path=invalid_json_file, schema=input_schema)

    if input_schema:
        assert str(excinfo.value) == log_message
    else:
        assert str(excinfo.value) == log_message.format(invalid_json_file)
