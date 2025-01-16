import copy
import importlib
import json
import re
from collections.abc import Callable
from contextlib import nullcontext as does_not_raise
from pathlib import Path

import jsonschema
import pytest

# from ethology.annotations.json_schemas import COCO_SCHEMA, VIA_SCHEMA
# from ethology.annotations.validators import (
#     # ValidCOCO,
#     # ValidJSON,
#     ValidVIA,
# )


@pytest.fixture()
def valid_via_schema() -> dict:
    """Return the valid VIA schema."""
    from ethology.annotations.json_schemas import VIA_SCHEMA

    return VIA_SCHEMA


@pytest.fixture()
def valid_coco_schema() -> dict:
    """Return the valid COCO schema."""
    from ethology.annotations.json_schemas import COCO_SCHEMA

    return COCO_SCHEMA


@pytest.fixture()
def valid_via_file_sample_1(annotations_test_data: dict) -> Path:
    """Return path to a valid VIA JSON file."""
    return annotations_test_data["VIA_JSON_sample_1.json"]


@pytest.fixture()
def valid_coco_file_sample_1(annotations_test_data: dict) -> Path:
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
def via_file_schema_mismatch(
    valid_via_file_sample_1: Path,
    tmp_path: Path,
) -> Path:
    """Return path to a VIA JSON file that does not match its schema.

    Specifically, we modify the type of the "width" of the first bounding box
    in the first image, from "int" to "float"
    """
    # Read valid JSON file
    with open(valid_via_file_sample_1) as f:
        data = json.load(f)

    # Modify file so that it doesn't match the corresponding schema
    _, img_dict = list(data["_via_img_metadata"].items())[0]
    img_dict["regions"][0]["shape_attributes"]["width"] = 49.5

    # Save the modified JSON to a new file
    out_json = tmp_path / f"{valid_via_file_sample_1.stem}_schema_error.json"
    with open(out_json, "w") as f:
        json.dump(data, f)
    return out_json


@pytest.fixture()
def coco_file_schema_mismatch(
    valid_coco_file_sample_1: Path,
    tmp_path: Path,
) -> Path:
    """Return path to a COCO JSON file that doesn't match its schema.

    Specifically, we modify the type of the object under the "annotations"
    key from "list of dicts" to "list of lists"
    """
    # Read valid JSON file
    with open(valid_coco_file_sample_1) as f:
        data = json.load(f)

    # Modify file so that it doesn't match the corresponding schema
    data["annotations"] = [[d] for d in data["annotations"]]

    # save the modified json to a new file
    out_json = tmp_path / f"{valid_coco_file_sample_1.stem}_schema_error.json"
    with open(out_json, "w") as f:
        json.dump(data, f)
    return out_json


@pytest.fixture()
def via_file_sample_1_with_missing_keys(
    valid_via_file_sample_1: Path, tmp_path: Path
) -> Callable:
    """Get paths to a modified VIA JSON 1 file with some required keys missing.

    This fixture is a factory of fixtures. It returns a function that can be
    used to create a fixture representing a VIA JSON file with some
    user-defined keys missing.

    Specifically, the fixture obtained is a tuple with:
    - the path to the VIA JSON file 1 modified to omit some keys, and
    - a dictionary holding the names of the images whose data was removed.
    """

    def _via_json_1_file_with_missing_keys(
        required_keys_to_pop: dict,
    ) -> tuple[Path, dict]:
        # Read valid json file
        with open(valid_via_file_sample_1) as f:
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
        out_json = (
            tmp_path / f"{valid_via_file_sample_1.stem}_missing_keys.json"
        )
        with open(out_json, "w") as f:
            json.dump(data, f)
        return out_json, edited_image_dicts

    return _via_json_1_file_with_missing_keys


@pytest.fixture()
def coco_file_sample_1_with_missing_keys(
    valid_coco_file_sample_1: Path, tmp_path: Path
) -> Callable:
    """Get path to a modified COCO JSON file with some required keys missing.

    This fixture is a factory of fixtures. It returns a function that can be
    used to create a fixture representing a COCO JSON file with some
    user-defined keys missing.

    Specifically, the fixture obtained is a tuple with:
    - the path to the COCO JSON 1 file modified to omit some keys, and
    - a dictionary holding the names of the images whose data was removed.
    """

    def _coco_json_1_file_with_missing_keys(
        required_keys_to_pop: dict,
    ) -> tuple[Path, dict]:
        # Read valid json file
        with open(valid_coco_file_sample_1) as f:
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
            tmp_path / f"{valid_coco_file_sample_1.stem}_missing_keys.json"
        )
        with open(out_json, "w") as f:
            json.dump(data, f)
        return out_json, edited_image_dicts

    return _coco_json_1_file_with_missing_keys


@pytest.fixture()
def invalid_VIA_schema() -> dict:
    """Return an invalid VIA schema.

    Note: typos in the schema keywords are not considered violations
    and are simply interpreted as additional keys.
    """
    from ethology.annotations.json_schemas import VIA_SCHEMA

    invalid_VIA_schema = copy.deepcopy(VIA_SCHEMA)
    invalid_VIA_schema["type"] = "FOO"  # invalid value for a valid keyword
    return invalid_VIA_schema


@pytest.fixture()
def invalid_COCO_schema() -> dict:
    """Return an invalid COCO schema.

    Note: typos in the schema keywords are not considered violations
    and are simply interpreted as additional keys.
    """
    from ethology.annotations.json_schemas import COCO_SCHEMA

    invalid_COCO_schema = copy.deepcopy(COCO_SCHEMA)
    invalid_COCO_schema["properties"]["images"]["type"] = 123
    # "type" must be a string according to the meta-schema
    return invalid_COCO_schema


@pytest.mark.parametrize(
    "input_file, input_schema",
    [
        ("VIA_JSON_sample_1.json", None),
        ("VIA_JSON_sample_2.json", "valid_via_schema"),
        ("COCO_JSON_sample_1.json", None),
        ("COCO_JSON_sample_2.json", "valid_coco_schema"),
    ],
)
def test_valid_json(
    input_file: str,
    input_schema: dict | None,
    annotations_test_data: dict,
    request: pytest.FixtureRequest,
):
    """Test the ValidJSON validator with valid inputs."""
    filepath = annotations_test_data[input_file]
    if input_schema:
        input_schema = request.getfixturevalue(input_schema)

    with does_not_raise():
        from ethology.annotations.validators import ValidJSON
        ValidJSON(
            path=filepath,
            schema=input_schema,
        )


@pytest.mark.parametrize(
    "invalid_input_file, input_schema, expected_exception, log_message",
    [
        (
            "json_file_decode_error",
            None,
            pytest.raises(ValueError),
            "Error decoding JSON data from file",  # decoding error
        ),
        (
            "json_file_not_found_error",
            None,
            pytest.raises(FileNotFoundError),
            "File not found",  # file error
        ),
        (
            "json_file_decode_error",
            "valid_via_schema",  # this error should be independent of the schema
            pytest.raises(ValueError),
            "Error decoding JSON data from file",  # decoding error
        ),
        (
            "json_file_not_found_error",
            "valid_coco_schema",  # this error should be independent of the schema
            pytest.raises(FileNotFoundError),
            "File not found",  # file error
        ),
        (
            "via_file_schema_mismatch",
            "valid_via_schema",
            pytest.raises(jsonschema.exceptions.ValidationError),
            "49.5 is not of type 'integer'\n\n",  # schema mismatch
        ),
        (
            "coco_file_schema_mismatch",
            "valid_coco_schema",
            pytest.raises(jsonschema.exceptions.ValidationError),
            "[{'area': 432, 'bbox': [1278, 556, 16, 27], 'category_id': 1, "
            "'id': 8917, 'image_id': 199, 'iscrowd': 0}] is not of type "
            "'object'\n\n",  # schema mismatch
        ),
    ],
)
def test_valid_json_invalid_inputs(
    invalid_input_file: str,
    input_schema: str | None,
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
    if input_schema:
        input_schema = request.getfixturevalue(input_schema)

    with expected_exception as excinfo:
        from ethology.annotations.validators import ValidJSON
        ValidJSON(path=invalid_json_file, schema=input_schema)

    # Check that the error message contains expected string
    assert log_message in str(excinfo.value)

    # If error is not a schema-mismatch, check the error message contains
    # file path
    if not isinstance(excinfo.value, jsonschema.exceptions.ValidationError):
        assert invalid_json_file.name in str(excinfo.value)


@pytest.mark.parametrize(
    "input_file, invalid_schema",
    [
        ("valid_via_file_sample_1", "invalid_VIA_schema"),
        ("valid_via_file_sample_1", "invalid_VIA_schema"),
        ("valid_coco_file_sample_1", "invalid_COCO_schema"),
        ("valid_coco_file_sample_1", "invalid_COCO_schema"),
    ],
)
def test_valid_json_invalid_schema(input_file, invalid_schema, request):
    """Test the ValidJSON validator throws an error when the schema is
    invalid.
    """
    input_file = request.getfixturevalue(input_file)
    invalid_schema = request.getfixturevalue(invalid_schema)

    with pytest.raises(jsonschema.exceptions.SchemaError) as excinfo:
        from ethology.annotations.validators import ValidJSON
        ValidJSON(
            path=input_file,
            schema=invalid_schema,
        )

    # Check the error message is as expected
    assert "is not valid under any of the given schemas" in str(excinfo.value)


# @pytest.mark.parametrize(
#     "input_file, validator",
#     [
#         # ("VIA_JSON_sample_1.json", ValidVIA),
#         # ("VIA_JSON_sample_2.json", ValidVIA),
#         ("COCO_JSON_sample_1.json", ValidCOCO),
#         ("COCO_JSON_sample_2.json", ValidCOCO),
#     ],
# )
# def test_valid_via_and_coco(
#     input_file: str, validator: Callable, annotations_test_data: dict
# ):
#     """Test the file-specific validators (VIA or COCO) validators with valid
#     inputs.
#     """
#     filepath = annotations_test_data[input_file]
#     with does_not_raise():
#         validator(path=filepath)


# @pytest.mark.parametrize(
#     "invalid_input_file, validator, expected_exception, log_message",
#     [
#         (
#             "json_file_decode_error",
#             ValidVIA,
#             pytest.raises(ValueError),
#             "Error decoding JSON data from file",
#         ),
#         (
#             "json_file_not_found_error",
#             ValidCOCO,
#             pytest.raises(FileNotFoundError),
#             "File not found",
#         ),
#         (
#             "via_file_schema_mismatch",
#             ValidVIA,
#             pytest.raises(jsonschema.exceptions.ValidationError),
#             "49.5 is not of type 'integer'",
#         ),
#         (
#             "coco_file_schema_mismatch",
#             ValidCOCO,
#             pytest.raises(jsonschema.exceptions.ValidationError),
#             "[{'area': 432, 'bbox': [1278, 556, 16, 27], 'category_id': 1, "
#             "'id': 8917, 'image_id': 199, 'iscrowd': 0}] is not of type "
#             "'object'\n\n",
#         ),
#     ],
# )
# def test_valid_via_and_coco_invalid_inputs(
#     invalid_input_file: str,
#     validator: Callable,
#     expected_exception: pytest.raises,
#     log_message: str,
#     request: pytest.FixtureRequest,
# ):
#     """Test the file-specific validators (VIA or COCO) throw the expected
#     errors when passed invalid inputs.

#     The invalid inputs cases covered in this test are:
#     - a JSON file that cannot be decoded
#     - a JSON file that does not exist
#     - a JSON file that does not match the given (correct) schema
#     """
#     invalid_json_file = request.getfixturevalue(invalid_input_file)

#     # Check the file-specific validator throws an error for the
#     # default schema
#     with expected_exception as excinfo:
#         validator(path=invalid_json_file)

#     # Check that the error message contains expected string
#     assert log_message in str(excinfo.value)

#     # Check the error message contains file path
#     # assert invalid_json_file.name in str(excinfo.value)
#     if not isinstance(excinfo.value, jsonschema.exceptions.ValidationError):
#         assert invalid_json_file.name in str(excinfo.value)


@pytest.mark.parametrize(
    "input_file, valid_schema_str, invalid_schema",
    [
        (
            "valid_via_file_sample_1",
            # ValidVIA,
            "VIA_SCHEMA",
            "invalid_VIA_schema",
        ),
        (
            "valid_via_file_sample_1",
            # ValidVIA,
            "VIA_SCHEMA",
            "invalid_VIA_schema",
        ),
    ],
)
def test_valid_via_and_coco_invalid_schema(
    input_file,
    valid_schema_str,
    invalid_schema,
    request,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test the file-specific validators (VIA and COCO) throw an error when
    the schema is invalid.
    """
    input_file = request.getfixturevalue(input_file)
    invalid_schema = request.getfixturevalue(invalid_schema)

    monkeypatch.setattr(
        f"ethology.annotations.json_schemas.{valid_schema_str}", invalid_schema
    )

    with pytest.raises(jsonschema.exceptions.SchemaError) as excinfo:
        # from ethology.annotations.validators import ValidVIA
        validator_module = importlib.import_module("ethology.annotations.validators")

        validator_module.ValidVIA(path=input_file)

    # Check the error message is as expected
    assert "is not valid under any of the given schemas" in str(excinfo.value)


@pytest.mark.parametrize(
    ("expected_missing_keys, log_message"),
    [
        (
            {"main": ["_via_image_id_list"]},
            "Required key(s) ['_via_image_id_list'] not found.",
        ),
        (
            {"main": ["_via_image_id_list", "_via_img_metadata"]},
            "Required key(s) ['_via_image_id_list', '_via_img_metadata'] "
            "not found.",
        ),
        (
            {"image_keys": ["filename"]},
            "Required key(s) ['filename'] not found " "for {}.",
        ),
        (
            {"region_keys": ["shape_attributes"]},
            "Required key(s) ['shape_attributes'] not found "
            "for region 0 under {}.",
        ),
        (
            {"shape_attributes_keys": ["x"]},
            "Required key(s) ['x'] not found for region 0 under {}.",
        ),
    ],
)
def test_valid_via_missing_keys_in_file(
    via_file_sample_1_with_missing_keys: Callable,
    expected_missing_keys: dict,
    log_message: str,
):
    """Test the ValidVIA validator throws an error when the
    input misses some required keys.
    """
    # Create an invalid VIA JSON file with missing keys
    invalid_json_file, edited_image_dicts = (
        via_file_sample_1_with_missing_keys(
            expected_missing_keys,  # required keys to remove
        )
    )

    # Get key of image whose data has been edited
    # (if the modified data belongs to the "main" section of the VIA or
    # COCO JSON file, the key for the modified image is None)
    modified_data = list(expected_missing_keys.keys())[0]
    img_key = edited_image_dicts.get(modified_data, None)

    # Run validation
    with pytest.raises(ValueError) as excinfo:
        from ethology.annotations.validators import ValidVIA
        ValidVIA(
            path=invalid_json_file,
        )

    # Check the error message is as expected.
    # If the modified data belongs to a specific image, its key should
    # appear in the error message.
    assert str(excinfo.value) == log_message.format(img_key)


@pytest.mark.parametrize(
    "keys2remove",
    [
        {"main": ["_via_image_id_list"]},
        {"images": ["regions"]},
        {"regions": ["region_attributes"]},
        {"shape_attributes": ["width", "height"]},
    ],
)
def test_valid_via_missing_keys_in_schema(
    valid_via_file_sample_1: Path,
    keys2remove: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test the ValidVIA validator throws an error when the schema
    does not include a required key.
    """
    from ethology.annotations.json_schemas import VIA_SCHEMA

    # Set a property dict in the schema as the base_dict
    base_dict = VIA_SCHEMA["properties"]["_via_img_metadata"][  # type: ignore
        "additionalProperties"
    ]["properties"]

    # Map keys in `keys2remove` to subdictionaries in the schema
    map_key_to_subdict = {
        "main": VIA_SCHEMA["properties"],
        "images": base_dict,
        "regions": base_dict["regions"]["items"]["properties"],
        "shape_attributes": base_dict["regions"]["items"]["properties"][
            "shape_attributes"
        ]["properties"],
    }

    # Monkeypatch VIA_schema to define a schema with missing keys
    for ky in keys2remove:
        for ky_subdict in keys2remove[ky]:
            monkeypatch.delitem(
                map_key_to_subdict[ky],
                ky_subdict,
            )

    # Run validation
    with pytest.raises(ValueError) as excinfo:
        ValidVIA(path=valid_via_file_sample_1)

    # Check the error message is as expected
    # The regexp matches any full key paths that start with the expected keys
    # to remove
    missing_keys_pattern = "|".join(sorted(*keys2remove.values()))
    pattern = re.compile(
        rf"Required key\(s\) \[.*({missing_keys_pattern}).*\] "
        rf"not found in schema\."
    )
    assert re.match(pattern, str(excinfo.value))


@pytest.mark.parametrize(
    ("expected_missing_keys, log_message"),
    [
        (
            {"main": ["categories"]},
            "Required key(s) ['categories'] not found.",
        ),
        (
            {"main": ["categories", "images"]},
            "Required key(s) ['categories', 'images'] not found.",
        ),
        (
            {"image_keys": ["file_name"]},
            "Required key(s) ['file_name'] not found " "for image {}.",
        ),
        (
            {"annotations_keys": ["category_id"]},
            "Required key(s) ['category_id'] not found " "for annotation {}.",
        ),
        (
            {"categories_keys": ["id"]},
            "Required key(s) ['id'] not found " "for category {}.",
        ),
    ],
)
def test_valid_coco_missing_keys_in_file(
    coco_file_sample_1_with_missing_keys: Callable,
    expected_missing_keys: dict,
    log_message: str,
):
    """Test the ValidCOCO validator throws an error when the
    input misses some required keys.
    """
    # Create an invalid VIA JSON file with missing keys
    invalid_json_file, edited_image_dicts = (
        coco_file_sample_1_with_missing_keys(
            expected_missing_keys  # required keys to remove
        )
    )

    # Get key of image whose data has been edited
    # (if the modified data belongs to the "main" section of the VIA or
    # COCO JSON file, the key for the modified image is None)
    modified_data = list(expected_missing_keys.keys())[0]
    img_key = edited_image_dicts.get(modified_data, None)

    # Run validation
    with pytest.raises(ValueError) as excinfo:
        from ethology.annotations.validators import ValidCOCO
        ValidCOCO(
            path=invalid_json_file,
        )

    # Check the error message is as expected.
    # If the modified data belongs to a specific image, its key should
    # appear in the error message.
    assert str(excinfo.value) == log_message.format(img_key)


@pytest.mark.parametrize(
    "keys2remove",
    [
        {"main": ["images", "annotations"]},
        {"images": ["file_name"]},
        {"annotations": ["id"]},
        {"categories": ["supercategory", "name"]},
    ],
)
def test_valid_coco_missing_keys_in_schema(
    valid_coco_file_sample_1: Path,
    valid_coco_schema,
    keys2remove: dict,
    monkeypatch: pytest.MonkeyPatch,
):
    """Test the ValidCOCO validator throws an error when the schema
    does not include a required key.
    """
    # Redefine valid COCO_schema to point to a schema with missing keys
    for ky in keys2remove:
        if ky == "main":
            for ky_subdict in keys2remove[ky]:
                monkeypatch.delitem(
                    valid_coco_schema["properties"], ky_subdict
                )
        else:
            for ky_subdict in keys2remove[ky]:
                monkeypatch.delitem(
                    valid_coco_schema["properties"][ky]["items"]["properties"],  # type: ignore
                    ky_subdict,
                )

    # Monkeypatch the COCO schema to define a schema with missing keys
    # monkeypatch.setattr(f"ethology.annotations.json_schemas.COCO_SCHEMA", invalid_schema)

    # Run validation
    with pytest.raises(ValueError) as excinfo:
        from ethology.annotations.validators import ValidCOCO

        ValidCOCO(path=valid_coco_file_sample_1)

    # Check the error message is as expected
    # The regexp matches any keys that start as the expected keys to remove
    missing_keys_pattern = "|".join(sorted(*keys2remove.values()))
    pattern = re.compile(
        rf"Required key\(s\) \[.*({missing_keys_pattern}).*\] "
        rf"not found in schema\."
    )
    assert re.match(pattern, str(excinfo.value))


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
    from ethology.annotations.validators import _check_keys

    with expected_exception as excinfo:
        _check_keys(required_keys, input_data, expected_in_log_message)

    # If an exception is raised, check the error message is as expected
    if excinfo:
        missing_keys = set(required_keys) - input_data.keys()
        assert str(excinfo.value) == (
            f"Required key(s) {sorted(missing_keys)} "
            f"not found{expected_in_log_message}."
        )


@pytest.mark.parametrize(
    "input_schema, expected_properties_keys",
    [
        (
            "valid_via_schema",
            [
                "_via_attributes",
                "_via_attributes/file",
                "_via_attributes/region",
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
            "valid_coco_schema",
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
    input_schema: dict,
    expected_properties_keys: list,
    request: pytest.FixtureRequest,
):
    """Test the _extract_properties_keys helper function."""
    from ethology.annotations.validators import _extract_properties_keys

    input_schema = request.getfixturevalue(input_schema)

    list_keys = _extract_properties_keys(input_schema)

    assert list_keys == sorted(expected_properties_keys)
