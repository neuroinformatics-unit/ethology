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
                raise val_err
            except jsonschema.exceptions.SchemaError as schema_err:
                raise schema_err


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

    # Required keys for VIA JSON files
    required_keys: dict = {
        "main": ["_via_img_metadata", "_via_image_id_list"],
        "images": ["filename", "regions"],
        "regions": ["shape_attributes", "region_attributes"],
        "shape_attributes": ["x", "y", "width", "height"],
    }

    @path.validator
    def _file_contains_required_keys(self, attribute, value):
        """Ensure that the VIA JSON file contains the required keys."""
        # Read data as dict
        with open(value) as file:
            data = json.load(file)

        # Check first level keys
        _check_keys(self.required_keys["main"], data)

        # Check keys in nested dicts
        for img_str, img_dict in data["_via_img_metadata"].items():
            # Check keys for each image dictionary
            _check_keys(
                self.required_keys["images"],
                img_dict,
                additional_message=f" for {img_str}",
            )
            # Check keys for each region in an image
            for i, region in enumerate(img_dict["regions"]):
                # Check keys under first level per region
                _check_keys(
                    self.required_keys["regions"],
                    region,
                    additional_message=f" for region {i} under {img_str}",
                )

                # Check keys under "shape_attributes" per region
                _check_keys(
                    self.required_keys["shape_attributes"],
                    region["shape_attributes"],
                    additional_message=f" for region {i} under {img_str}",
                )

    @schema.validator
    def _schema_contains_required_keys(
        self, attribute, value
    ):  # REFACTOR with coco version?-------------
        """Ensure that the schema includes the required keys."""
        missing_keys = []

        # Get keys of "property" dicts in schema
        property_keys_in_schema = _extract_properties_keys(value)

        # --------- Factor out?
        # Prepare list of required "property" keys with full paths
        map_required_to_property_keys = {
            "main": "",
            "images": "_via_img_metadata",
            "regions": "_via_img_metadata/regions",
            "shape_attributes": "_via_img_metadata/regions/shape_attributes",
        }
        required_property_keys = []
        for ky, values in self.required_keys.items():
            for val in values:
                if ky == "main":
                    required_property_keys.append(val)
                else:
                    required_property_keys.append(
                        f"{map_required_to_property_keys[ky]}/{val}"
                    )
        # --------------------------------------------

        # Get list of keys that are required but not in schema
        missing_keys = set(required_property_keys) - set(
            property_keys_in_schema
        )

        # Raise error if there are missing keys in the schema
        if missing_keys:
            raise ValueError(
                f"Required key(s) {sorted(missing_keys)} not found in schema."
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

        def _singularise(key):
            # Helper function to singularise the input key
            return key[:-1] if key != "categories" else key[:-3] + "y"

        # Read file as dict
        with open(value) as file:
            data = json.load(file)

        # Check first level keys
        _check_keys(self.required_keys["main"], data)

        # Check keys in nested list of dicts
        for ky in ["images", "annotations", "categories"]:
            for instance_dict in data[ky]:
                _check_keys(
                    self.required_keys[ky],
                    instance_dict,
                    additional_message=(
                        f" for {_singularise(ky)} {instance_dict}"
                    ),
                )

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
                f"Required key(s) {sorted(missing_keys)} not found "
                "in schema."
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
    # The "property keys" are either "properties" or "additionalProperties"
    # as they are the keys with the relevant data
    property_keys = ["properties", "additionalProperties"]

    def _contains_properties_key(input: dict):
        """Return True if the input dictionary contains a property key."""
        return any(x in input for x in property_keys)

    def _get_properties_subdict(input: dict):
        """Get the subdictionary under the property key."""
        return input[next(k for k in property_keys if k in input)]

    keys_of_properties_dicts = []
    if "type" in input_dict:
        if _contains_properties_key(input_dict):
            # Get the subdictionary under the properties key
            properties_subdict = _get_properties_subdict(input_dict)

            # Check if there is a nested "properties" dict inside the current
            # one. If so, go down one level.
            if _contains_properties_key(properties_subdict):
                properties_subdict = _get_properties_subdict(
                    properties_subdict
                )

            # Add keys of deepest "properties dict" to list
            keys_of_properties_dicts.extend(
                [
                    f"{parent_key}/{ky}" if parent_key else ky
                    for ky in properties_subdict
                ]
            )

            # Inspect non-properties dictionaries under this properties subdict
            for ky, val in properties_subdict.items():
                full_key = f"{parent_key}/{ky}" if parent_key else ky
                keys_of_properties_dicts.extend(
                    _extract_properties_keys(val, full_key)
                )

        elif "items" in input_dict:
            # Analyse the dictionary under the "items" key
            properties_subdict = input_dict["items"]
            keys_of_properties_dicts.extend(
                _extract_properties_keys(
                    properties_subdict, parent_key=parent_key
                )
            )

    return sorted(keys_of_properties_dicts)
