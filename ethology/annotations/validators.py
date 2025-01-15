"""Validators for supported annotation files."""

import json
from pathlib import Path

import attrs
import jsonschema
import jsonschema.exceptions
import jsonschema.validators
from attrs import define, field, validators

from ethology.annotations.json_schemas import COCO_SCHEMA, VIA_SCHEMA


@define
class ValidJSON:
    """Class for valid JSON files.

    It checks the JSON file exists, can be decoded, and optionally
    validates the file against a JSON schema.

    Attributes
    ----------
    path : pathlib.Path
        Path to the JSON file.

    schema : dict, optional
        JSON schema to validate the file against.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the JSON file cannot be decoded.
    jsonschema.exceptions.ValidationError
        If the type of any of the keys in the JSON file
        does not match the type specified in the schema.


    Notes
    -----
    https://json-schema.org/understanding-json-schema/

    """

    # Required attribute
    path: Path = field(validator=validators.instance_of(Path))

    # Optional attribute
    schema: dict | None = field(default=None)

    @path.validator
    def _file_is_json(self, attribute, value):
        """Ensure that the file is a JSON file."""
        try:
            with open(value) as file:
                json.load(file)
        except FileNotFoundError as not_found_error:
            raise FileNotFoundError(
                f"File not found: {value}."
            ) from not_found_error
        except json.JSONDecodeError as decode_error:
            raise ValueError(
                f"Error decoding JSON data from file: {value}."
            ) from decode_error

    @path.validator
    def _file_matches_JSON_schema(self, attribute, value):
        """Ensure that the JSON file matches the expected schema.

        The schema validation only checks the type for each specified
        key if the key exists. It does not check that the keys in the
        schema are present in the JSON file.
        """
        # read json file
        with open(value) as file:
            data = json.load(file)

        # check against schema if provided
        if self.schema:
            try:
                jsonschema.validate(instance=data, schema=self.schema)
            except jsonschema.exceptions.ValidationError as val_err:
                # forward the error message as it is quite informative
                raise val_err


@define
class ValidVIA(ValidJSON):
    """Class for valid VIA JSON files.

    It checks the input file is a `ValidJSON` and additionally checks the
    file contains the required keys.

    Attributes
    ----------
    path : pathlib.Path
        Path to the VIA JSON file.

    schema : dict
        The JSON schema is set to VIA_SCHEMA.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the JSON file cannot be decoded.
    jsonschema.exceptions.ValidationError
        If the type of any of the keys in the JSON file
        does not match the type specified in the schema.
    ValueError
        If the VIA JSON file misses any of the required keys.

    """

    # Run the parent's validator on the provided path
    path: Path = field(validator=attrs.fields(ValidJSON).path.validator)

    # Run the parent's validator on the hardcoded schema
    schema: dict = field(
        validator=attrs.fields(ValidJSON).schema.validator,  # type: ignore
        default=VIA_SCHEMA,
        init=False,
        # init=False makes the attribute to be unconditionally initialized
        # with the specified default
    )

    # TODO: add a validator to check the schema defines types
    # for the required keys

    # run additional validators from this class
    @path.validator
    def _file_contains_required_keys(self, attribute, value):
        """Ensure that the VIA JSON file contains the required keys."""
        required_keys = {
            "main": ["_via_img_metadata", "_via_image_id_list"],
            "images_keys": ["filename", "regions"],
            "region_keys": ["shape_attributes", "region_attributes"],
            "shape_attributes_keys": ["x", "y", "width", "height"],
        }

        # Read data as dict
        with open(value) as file:
            data = json.load(file)

        # Check first level keys
        _check_keys(required_keys["main"], data)

        # Check keys in nested dicts
        for img_str, img_dict in data["_via_img_metadata"].items():
            # Check keys for each image dictionary
            _check_keys(
                required_keys["images_keys"],
                img_dict,
                additional_message=f" for {img_str}",
            )
            # Check keys for each region
            for i, region in enumerate(img_dict["regions"]):
                _check_keys(
                    required_keys["region_keys"],
                    region,
                    additional_message=f" for region {i} under {img_str}",
                )

                # Check keys under shape_attributes
                _check_keys(
                    required_keys["shape_attributes_keys"],
                    region["shape_attributes"],
                    additional_message=f" for region {i} under {img_str}",
                )


@define
class ValidCOCO(ValidJSON):
    """Class valid COCO JSON files for untracked data.

    It checks the input COCO JSON file contains the required keys.

    Attributes
    ----------
    path : pathlib.Path
        Path to the COCO JSON file.

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If the JSON file cannot be decoded.
    jsonschema.exceptions.ValidationError
        If the type of any of the keys in the JSON file
        does not match the type specified in the schema.
    ValueError
        If the COCO JSON file misses any of the required keys.

    """

    # Run the parent's validators on the provided path
    path: Path = field(validator=attrs.fields(ValidJSON).path.validator)

    # Run the parent's validator on the hardcoded schema
    schema: dict = field(
        validator=attrs.fields(ValidJSON).schema.validator,  # type: ignore
        default=COCO_SCHEMA,
        init=False,
        # init=False makes the attribute to be unconditionally initialized
        # with the specified default
    )

    # Required keys for COCO JSON files
    required_keys: dict = {
        "main": ["images", "annotations", "categories"],
        "images": ["id", "file_name"],
        "annotations": ["id", "image_id", "bbox", "category_id"],
        "categories": ["id", "name", "supercategory"],
    }

    @path.validator
    def _file_contains_required_keys(self, attribute, value):
        """Ensure that the COCO JSON file contains the required keys."""

        def _check_nested_dicts(data, key):
            # Helper function to singularise the input key
            def _singularise(key):
                return key[:-1] if key != "categories" else key[:-3] + "y"

            # Check keys for each instance in the list
            for instance_dict in data[key]:
                _check_keys(
                    self.required_keys[key],
                    instance_dict,
                    additional_message=(
                        f" for {_singularise(key)} {instance_dict}"
                    ),
                )

        # Read file as dict
        with open(value) as file:
            data = json.load(file)

        # Check first level keys
        _check_keys(self.required_keys["main"], data)

        # Check keys in nested list of dicts
        for ky in ["images", "annotations", "categories"]:
            _check_nested_dicts(data, ky)

    @schema.validator
    def _schema_contains_required_keys(self, attribute, value):
        """Ensure that the schema includes the required keys."""
        missing_keys = []

        # Get keys of properties dicts in schema
        properties_keys_in_schema = _extract_properties_keys(value)

        # Prepare list of required "properties" keys with full paths
        required_properties_keys = [
            f"{level}/{ky}" if level != "main" else ky
            for level, required_keys in self.required_keys.items()
            for ky in required_keys
        ]

        # Get list of keys that are required but not in schema
        missing_keys = set(required_properties_keys) - set(
            properties_keys_in_schema
        )

        # Raise error if there are missing keys in the schema
        if missing_keys:
            raise ValueError(
                f"Required key(s) {sorted(missing_keys)} not found in schema."
            )


def _check_keys(
    list_required_keys: list[str],
    data_dict: dict,
    additional_message: str = "",
):
    """Check if the required keys are present in the input data_dict."""
    missing_keys = set(list_required_keys) - data_dict.keys()
    if missing_keys:
        raise ValueError(
            f"Required key(s) {sorted(missing_keys)} not "
            f"found{additional_message}."
        )


def _extract_properties_keys(input_dict: dict, parent_key="") -> list:
    """Recursively extract "properties" keys.

    Recursively extract keys of all subdictionaries in input_dict that are
    values to a "properties" key. The input nested_dict represents a JSON
    schema dictionary (see https://json-schema.org/understanding-json-schema/about).

    The "properties" key always appears as part of a dictionary with at least
    another key, that is "type" or "item".
    """
    keys_in_properties_dicts = []
    property_str_options = ["properties", "additionalProperties"]

    if "type" in input_dict:
        if any(x in input_dict for x in property_str_options):
            # Get the subdictionary under the properties key
            properties_subdict = input_dict[
                next(k for k in property_str_options if k in input_dict)
            ]

            # Check if there is a nested "properties" dict inside the current
            # one. If so, go down one level.
            if any(x in properties_subdict for x in property_str_options):
                properties_subdict = properties_subdict[
                    next(
                        k
                        for k in property_str_options
                        if k in properties_subdict
                    )
                ]

            # Add keys of deepest "properties dict" to list
            keys_in_properties_dicts.extend(
                [
                    f"{parent_key}/{ky}" if parent_key else ky
                    for ky in properties_subdict
                ]
            )

            # Inspect non-properties dictionaries under this properties subdict
            for ky, val in properties_subdict.items():
                full_key = f"{parent_key}/{ky}" if parent_key else ky
                keys_in_properties_dicts.extend(
                    _extract_properties_keys(val, full_key)
                )

        elif "items" in input_dict:
            # Analyse the dictionary under the "items" key
            properties_subdict = input_dict["items"]
            keys_in_properties_dicts.extend(
                _extract_properties_keys(
                    properties_subdict, parent_key=parent_key
                )
            )

    return sorted(keys_in_properties_dicts)
