from contextlib import nullcontext as does_not_raise

import pytest

from ethology.validators.annotations import ValidCOCO, ValidVIA
from ethology.validators.json_schemas.utils import (
    _check_required_keys_in_dict,
    _check_required_properties_keys,
    _extract_properties_keys,
)


@pytest.fixture()
def default_VIA_schema() -> dict:
    """Get default VIA schema."""
    from ethology.validators.json_schemas.utils import _get_default_schema

    return _get_default_schema("VIA")


@pytest.fixture()
def default_COCO_schema() -> dict:
    """Get default COCO schema."""
    from ethology.validators.json_schemas.utils import _get_default_schema

    return _get_default_schema("COCO")


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
