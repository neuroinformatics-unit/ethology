import json
from contextlib import nullcontext as does_not_raise

import jsonschema
import pytest

from ethology.annotations.json_schemas import (
    COCO_UNTRACKED_SCHEMA,
    VIA_UNTRACKED_SCHEMA,
)
from ethology.annotations.validators import (
    ValidJSON,
    ValidVIAUntrackedJSON,
)


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
def via_json_file_with_missing_keys(tmp_path, annotations_test_data):
    def _via_json_file_with_missing_keys(
        valid_json_filename, required_keys_to_pop
    ):
        """Return path to a JSON file that is missing required keys."""
        # read valid json file
        valid_json_path = annotations_test_data[valid_json_filename]
        with open(valid_json_path) as f:
            data = json.load(f)

        # remove any keys in the first level
        for key in required_keys_to_pop.get("main", []):
            data.pop(key)

        # remove keys in nested dicts
        for _, img_dict in data["_via_img_metadata"].items():
            # remove keys for each image dictionary
            for key in required_keys_to_pop.get("image_keys", []):
                img_dict.pop(key)

            for region in img_dict["regions"]:
                # remove keys for each region
                for key in required_keys_to_pop.get("region_keys", []):
                    region.pop(key)

                # remove keys under shape_attributes
                for key in required_keys_to_pop.get(
                    "shape_attributes_keys", []
                ):
                    region["shape_attributes"].pop(key)

        # save the modified json to a new file
        out_json = tmp_path / f"{valid_json_path.name}_missing_keys.json"
        with open(out_json, "w") as f:
            json.dump(data, f)
        return out_json

    return _via_json_file_with_missing_keys


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
            "49.5 is not of type 'integer'\n\n",
        ),
        (
            "coco_json_file_with_schema_error",
            COCO_UNTRACKED_SCHEMA,
            pytest.raises(jsonschema.exceptions.ValidationError),
            "[{'area': 432, 'bbox': [1278, 556, 16, 27], 'category_id': 1, "
            "'id': 8917, 'image_id': 199, 'iscrowd': 0}] is not of type "
            "'object'\n\n",
        ),
    ],
)
def test_valid_json_errors(
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
        assert log_message in str(excinfo.value)
    else:
        assert log_message.format(invalid_json_file) == str(excinfo.value)


@pytest.mark.parametrize(
    "input_json_file",
    [
        "VIA_JSON_sample_1.json",
        "VIA_JSON_sample_2.json",
    ],
)
def test_valid_via_untracked_json(annotations_test_data, input_json_file):
    filepath = annotations_test_data[input_json_file]
    with does_not_raise():
        ValidVIAUntrackedJSON(
            path=filepath,
        )


@pytest.mark.parametrize(
    "valid_json_file",
    [
        "VIA_JSON_sample_1.json",
        "VIA_JSON_sample_2.json",
    ],
)
@pytest.mark.parametrize(
    "missing_keys, expected_exception, log_message",
    [
        (
            {"main": ["_via_image_id_list"]},
            pytest.raises(ValueError),
            "Required key(s) {'_via_image_id_list'} not found "
            "in ['_via_settings', '_via_img_metadata', '_via_attributes', "
            "'_via_data_format_version'].",
        ),
        (
            {"image_keys": ["filename"]},
            pytest.raises(ValueError),
            "Required key(s) {'filename'} not found "
            "in ['size', 'regions', 'file_attributes'] "
            "for 09.08_09.08.2023-01-Left_frame_001764.png15086122.",
        ),
        (
            {"region_keys": ["shape_attributes"]},
            pytest.raises(ValueError),
            "The JSON data does not contain the required keys: annotations.",
        ),
        (
            {"shape_attributes_keys": ["x"]},
            pytest.raises(ValueError),
            "The JSON data does not contain the required keys: annotations.",
        ),
    ],
)
def test_valid_via_untracked_json_missing_keys(
    valid_json_file,
    missing_keys,
    via_json_file_with_missing_keys,
    expected_exception,
    log_message,
):
    # create invalid json file with missing keys
    invalid_json_file = via_json_file_with_missing_keys(
        valid_json_file, missing_keys
    )

    # run validatio
    with expected_exception as excinfo:
        ValidVIAUntrackedJSON(
            path=invalid_json_file,
        )

    assert str(excinfo.value) == log_message


# def test_valid_via_untracked_json ---> checks required keys
# def test_valid_coco_untracked_json ---> checks required keys
# def test_check_keys?
