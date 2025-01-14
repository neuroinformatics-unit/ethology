import json
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import jsonschema
import pytest

from ethology.annotations.json_schemas import COCO_SCHEMA, VIA_SCHEMA
from ethology.annotations.validators import (
    ValidCOCOJSON,
    ValidJSON,
    ValidVIAJSON,
    _check_keys,
)


@pytest.fixture()
def valid_via_json_1_file(annotations_test_data: dict) -> Path:
    """Return path to a valid VIA JSON file."""
    return annotations_test_data["VIA_JSON_sample_1.json"]


@pytest.fixture()
def valid_coco_json_1_file(annotations_test_data: dict) -> Path:
    """Return path to a valid COCO JSON file."""
    return annotations_test_data["COCO_JSON_sample_1.json"]


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
def via_json_file_schema_error(
    tmp_path: Path,
    annotations_test_data: dict,
) -> Path:
    """Return path to a VIA JSON file that doesn't match its schema.

    It uses the `_json_file_with_schema_error` factory function.
    """
    return _json_file_with_schema_error(
        tmp_path,
        annotations_test_data["VIA_JSON_sample_1.json"],
    )


@pytest.fixture()
def coco_json_file_schema_error(
    tmp_path: Path,
    annotations_test_data: dict,
) -> Path:
    """Return path to a COCO JSON file that doesn't match its schema.

    It uses the `_json_file_with_schema_error` factory function.
    """
    return _json_file_with_schema_error(
        tmp_path,
        annotations_test_data["COCO_JSON_sample_1.json"],
    )


def _json_file_with_schema_error(
    out_parent_path: Path, json_valid_path: Path
) -> Path:
    """From a valid input JSON file, return path to a JSON file that doesn't
    match the expected schema.
    """
    # Read valid JSON file
    with open(json_valid_path) as f:
        data = json.load(f)

    # Modify file so that it doesn't match the corresponding schema
    # if file is a VIA JSON test file, change "width" of a bounding box from
    # int to float
    if "VIA" in json_valid_path.name:
        _, img_dict = list(data["_via_img_metadata"].items())[0]
        img_dict["regions"][0]["shape_attributes"]["width"] = 49.5

    # if file is a COCO JSON test file, change "annotations" from list of
    # dicts to list of lists
    elif "COCO" in json_valid_path.name:
        data["annotations"] = [[d] for d in data["annotations"]]

    # save the modified json to a new file
    out_json = out_parent_path / f"{json_valid_path.name}_schema_error.json"
    with open(out_json, "w") as f:
        json.dump(data, f)
    return out_json


@pytest.fixture()
def via_json_1_file_with_missing_keys(
    valid_via_json_1_file: Path, tmp_path: Path
) -> Callable:
    """Get paths to a modified VIA JSON 1 file with some required keys missing.

    A VIA JSON file can be seen as a nested dictionary ("images" contain
    "regions" which contain "shape attributes"). Therefore the missing keys
    can be defined at the image level, the region level, or the shape attribute
    level.

    This fixture is a factory of fixtures. It returns a function that can be
    used to create a fixture that is a tuple with:
    - the path to the VIA JSON file 1 modified to omit some keys, and
    - a dictionary holding the names of the images whose data was removed.
    """

    def _via_json_1_file_with_missing_keys(
        required_keys_to_pop: dict,
    ) -> tuple[Path, dict]:
        """Return a tuple with:
        - the path to the VIA JSON file with some required keys missing,
        - a dictionary with the names of the images whose data was removed.
        """
        # Read valid json file
        with open(valid_via_json_1_file) as f:
            data = json.load(f)

        # Remove any keys in the first level
        for key in required_keys_to_pop.get("main", []):
            data.pop(key)

        # Remove any keys in nested dictionaries
        edited_image_dicts = {}
        if "_via_img_metadata" in data:
            list_img_metadata_tuples = list(data["_via_img_metadata"].items())

            # If image keys are specified in the keys to remove,
            # remove them in the first image dictionary that appears
            # in the VIA JSON file
            img_str, img_dict = list_img_metadata_tuples[0]
            edited_image_dicts["image_keys"] = img_str
            for key in required_keys_to_pop.get("image_keys", []):
                img_dict.pop(key)

            # If region keys are specified in the keys to remove,
            # remove them in the first region under the second
            # image dictionary that appears in the VIA JSON file
            img_str, img_dict = list_img_metadata_tuples[1]
            edited_image_dicts["region_keys"] = img_str
            for key in required_keys_to_pop.get("region_keys", []):
                img_dict["regions"][0].pop(key)

            # If shape attribute keys are specified in the keys to remove,
            # remove them in the first region under third image dictionary
            # that appears in the VIA JSON file
            img_str, img_dict = list_img_metadata_tuples[2]
            edited_image_dicts["shape_attributes_keys"] = img_str
            for key in required_keys_to_pop.get("shape_attributes_keys", []):
                img_dict["regions"][0]["shape_attributes"].pop(key)

        # Save the modified json to a new file
        out_json = tmp_path / f"{valid_via_json_1_file.name}_missing_keys.json"
        with open(out_json, "w") as f:
            json.dump(data, f)
        return out_json, edited_image_dicts

    return _via_json_1_file_with_missing_keys


@pytest.fixture()
def coco_json_1_file_with_missing_keys(
    valid_coco_json_1_file: Path, tmp_path: Path
) -> Callable:
    """Get path to a modified COCO JSON file with some required keys missing.

    A COCO JSON file can be seen as a dictionary that maps certain keys (such
    as "images", "annotations" or "categories") to lists of data. Therefore
    the missing keys can be defined for the data under "images", "annotations"
    or "categories".

    This fixture is a factory of fixtures. It returns a function that can be
    used to create a fixture that is a tuple with:
    - the path to the COCO JSON 1 file modified to omit some keys, and
    - a dictionary holding the names of the images whose data was removed.
    """

    def _coco_json_1_file_with_missing_keys(
        required_keys_to_pop: dict,
    ) -> tuple[Path, dict]:
        """Return path to a modified COCO JSON 1 file with some required
        keys missing.
        """
        # Read valid json file
        with open(valid_coco_json_1_file) as f:
            data = json.load(f)

        # Remove any keys in the first level
        for key in required_keys_to_pop.get("main", []):
            data.pop(key)

        # Remove required image keys in the first images dictionary
        edited_image_dicts = {}
        if "images" in data:
            edited_image_dicts["image_keys"] = data["images"][0]
            for key in required_keys_to_pop.get("image_keys", []):
                data["images"][0].pop(key)

        # Remove required annotations keys in the first annotations dictionary
        if "annotations" in data:
            edited_image_dicts["annotations_keys"] = data["annotations"][0]
            for key in required_keys_to_pop.get("annotations_keys", []):
                data["annotations"][0].pop(key)

        # Remove required categories keys in the first categories dictionary
        if "categories" in data:
            edited_image_dicts["categories_keys"] = data["categories"][0]
            for key in required_keys_to_pop.get("categories_keys", []):
                data["categories"][0].pop(key)

        # Save the modified json to a new file
        out_json = (
            tmp_path / f"{valid_coco_json_1_file.name}_missing_keys.json"
        )
        with open(out_json, "w") as f:
            json.dump(data, f)
        return out_json, edited_image_dicts

    return _coco_json_1_file_with_missing_keys


@pytest.mark.parametrize(
    "input_file_standard, input_schema",
    [
        ("VIA", None),
        ("VIA", VIA_SCHEMA),
        ("COCO", None),
        ("COCO", COCO_SCHEMA),
    ],
)
@pytest.mark.parametrize(
    "input_json_file_suffix",
    ["JSON_sample_1.json", "JSON_sample_2.json"],
)
def test_valid_json(
    input_file_standard: str,
    input_json_file_suffix: str,
    input_schema: dict | None,
    annotations_test_data: dict,
):
    """Test the ValidJSON validator with valid inputs."""
    filepath = annotations_test_data[
        f"{input_file_standard}_{input_json_file_suffix}"
    ]

    with does_not_raise():
        ValidJSON(
            path=filepath,
            schema=input_schema,
        )


@pytest.mark.parametrize(
    "invalid_json_fixture_name, input_schema, expected_exception, log_message",
    [
        (
            "json_file_decode_error",
            None,  # should be independent of schema
            pytest.raises(ValueError),
            "Error decoding JSON data from file",
        ),
        (
            "json_file_not_found_error",
            None,  # should be independent of schema
            pytest.raises(FileNotFoundError),
            "File not found",
        ),
        (
            "json_file_decode_error",
            VIA_SCHEMA,  # should be independent of schema
            pytest.raises(ValueError),
            "Error decoding JSON data from file",
        ),
        (
            "json_file_not_found_error",
            COCO_SCHEMA,  # should be independent of schema
            pytest.raises(FileNotFoundError),
            "File not found",
        ),
        (
            "via_json_file_schema_error",
            VIA_SCHEMA,
            pytest.raises(jsonschema.exceptions.ValidationError),
            "49.5 is not of type 'integer'\n\n",
        ),
        (
            "coco_json_file_schema_error",
            COCO_SCHEMA,
            pytest.raises(jsonschema.exceptions.ValidationError),
            "[{'area': 432, 'bbox': [1278, 556, 16, 27], 'category_id': 1, "
            "'id': 8917, 'image_id': 199, 'iscrowd': 0}] is not of type "
            "'object'\n\n",
        ),
    ],
)
def test_valid_json_errors(
    invalid_json_fixture_name: str,
    input_schema: dict | None,
    expected_exception: pytest.raises,
    log_message: str,
    request: pytest.FixtureRequest,
):
    """Test the ValidJSON validator throws the expected errors when given
    invalid inputs.
    """
    invalid_json_file = request.getfixturevalue(invalid_json_fixture_name)

    with expected_exception as excinfo:
        ValidJSON(path=invalid_json_file, schema=input_schema)

    # Check that the error message contains expected string
    assert log_message in str(excinfo.value)

    # If error is not related to JSON schema, check the error message contains
    # file path
    if not isinstance(excinfo.value, jsonschema.exceptions.ValidationError):
        assert invalid_json_file.name in str(excinfo.value)


@pytest.mark.parametrize(
    "input_json_file",
    [
        "VIA_JSON_sample_1.json",
        "VIA_JSON_sample_2.json",
    ],
)
def test_valid_via_json(annotations_test_data: dict, input_json_file: str):
    """Test the ValidVIAJSON validator with valid inputs."""
    filepath = annotations_test_data[input_json_file]
    with does_not_raise():
        ValidVIAJSON(
            path=filepath,
        )


@pytest.mark.parametrize(
    "expected_missing_keys, expected_exception, log_message",
    [
        (
            {"main": ["_via_image_id_list"]},
            pytest.raises(ValueError),
            "Required key(s) ['_via_image_id_list'] not found "
            "in ['_via_settings', '_via_img_metadata', '_via_attributes', "
            "'_via_data_format_version'].",
        ),
        (
            {"main": ["_via_image_id_list", "_via_img_metadata"]},
            pytest.raises(ValueError),
            "Required key(s) ['_via_image_id_list', '_via_img_metadata'] "
            "not found in ['_via_settings', '_via_attributes', "
            "'_via_data_format_version'].",
        ),
        (
            {"image_keys": ["filename"]},
            pytest.raises(ValueError),
            "Required key(s) ['filename'] not found "
            "in ['size', 'regions', 'file_attributes'] "
            "for {}.",
        ),
        (
            {"region_keys": ["shape_attributes"]},
            pytest.raises(ValueError),
            "Required key(s) ['shape_attributes'] not found in "
            "['region_attributes'] for region 0 under {}.",
        ),
        (
            {"shape_attributes_keys": ["x"]},
            pytest.raises(ValueError),
            "Required key(s) ['x'] not found in "
            "['name', 'y', 'width', 'height'] for region 0 under {}.",
        ),
    ],
)
def test_valid_via_json_missing_keys(
    expected_missing_keys: dict,
    expected_exception: pytest.raises,
    log_message: str,
    via_json_1_file_with_missing_keys: pytest.fixture,
):
    """Test the ValidVIAJSON validator throws an error when the input misses
    some required keys.
    """
    # Create an invalid VIA JSON file with missing keys
    invalid_json_file, edited_image_dicts = via_json_1_file_with_missing_keys(
        expected_missing_keys,  # required keys to remove
    )

    # Get key of image whose data has been edited
    # (if the modified data belongs to the "main" section of the VIA JSON
    # file, the key for the modified image is None)
    modified_data = list(expected_missing_keys.keys())[0]
    img_key_str = edited_image_dicts.get(modified_data, None)

    # Run validation
    with expected_exception as excinfo:
        ValidVIAJSON(
            path=invalid_json_file,
        )

    # Check the error message is as expected.
    # If the modified data belongs to a specific image, its key should
    # appear in the error message
    assert str(excinfo.value) == log_message.format(img_key_str)


@pytest.mark.parametrize(
    "expected_missing_keys, expected_exception, log_message",
    [
        (
            {"main": ["categories"]},
            pytest.raises(ValueError),
            "Required key(s) ['categories'] not found "
            "in ['annotations', 'images', 'info', 'licenses'].",
        ),
        (
            {"main": ["categories", "images"]},
            pytest.raises(ValueError),
            "Required key(s) ['categories', 'images'] not found "
            "in ['annotations', 'info', 'licenses'].",
        ),
        (
            {"image_keys": ["file_name"]},
            pytest.raises(ValueError),
            "Required key(s) ['file_name'] not found in "
            "['height', 'id', 'width'] for image dict {}.",
        ),
        (
            {"annotations_keys": ["category_id"]},
            pytest.raises(ValueError),
            "Required key(s) ['category_id'] not found in "
            "['area', 'bbox', 'id', 'image_id', 'iscrowd'] for "
            "annotation dict {}.",
        ),
        (
            {"categories_keys": ["id"]},
            pytest.raises(ValueError),
            "Required key(s) ['id'] not found in "
            "['name', 'supercategory'] for category dict {}.",
        ),
    ],
)
def test_valid_coco_json_missing_keys(
    expected_missing_keys: dict,
    expected_exception: pytest.raises,
    log_message: str,
    coco_json_1_file_with_missing_keys: pytest.fixture,
):
    """Test the ValidCOCOJSON validator throws an error when the input misses
    some required keys.
    """
    # Create an invalid COCO JSON file with missing keys
    invalid_json_file, edited_image_dicts = coco_json_1_file_with_missing_keys(
        expected_missing_keys,  # keys to remove from "COCO_JSON_sample_1.json"
    )

    # Get dict of image whose data has been edited
    # (if the modified data belongs to the "main" section of the COCO JSON
    # file, the key for the modified image is None)
    modified_data = list(expected_missing_keys.keys())[0]
    img_dict = edited_image_dicts.get(modified_data, None)

    # Run validation
    with expected_exception as excinfo:
        ValidCOCOJSON(
            path=invalid_json_file,
        )

    # Check the error message is as expected.
    # If the modified data belongs to a specific image, its dict should
    # appear in the error message
    assert str(excinfo.value) == log_message.format(img_dict)


@pytest.mark.parametrize(
    "required_keys, input_data, expected_exception, expected_in_log_message",
    [
        (
            ["images", "annotations", "categories"],
            {"images": "", "annotations": "", "categories": ""},
            does_not_raise(),
            "",
        ),  # zero missing keys
        (
            ["images", "annotations", "categories"],
            {"annotations": "", "categories": ""},
            pytest.raises(ValueError),
            "",
        ),  # one missing key
        (
            ["images", "annotations", "categories"],
            {"annotations": ""},
            pytest.raises(ValueError),
            "",
        ),  # two missing keys
        (
            ["images", "annotations", "categories"],
            {"annotations": "", "categories": ""},
            pytest.raises(ValueError),
            "FOO",
        ),  # one missing key with additional message
    ],
)
def test_check_keys(
    required_keys: list,
    input_data: dict,
    expected_exception: pytest.raises,
    expected_in_log_message: str,
):
    """Test the _check_keys helper function."""
    with expected_exception as excinfo:
        _check_keys(required_keys, input_data, expected_in_log_message)

    # If an exception is raised, check the error message is as expected
    if excinfo:
        missing_keys = set(required_keys) - input_data.keys()
        assert str(excinfo.value) == (
            f"Required key(s) {sorted(missing_keys)} not "
            f"found in {list(input_data.keys())}{expected_in_log_message}."
        )
