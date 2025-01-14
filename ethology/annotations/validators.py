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
        "images_keys": ["id", "file_name"],
        "annotations_keys": ["id", "image_id", "bbox", "category_id"],
        "categories_keys": ["id", "name", "supercategory"],
    }

    @path.validator
    def _file_contains_required_keys(self, attribute, value):
        """Ensure that the COCO JSON file contains the required keys."""
        # Read data as dict
        with open(value) as file:
            data = json.load(file)

        # Check first level keys
        _check_keys(self.required_keys["main"], data)

        # Check keys in images dicts
        for img_dict in data["images"]:
            _check_keys(
                self.required_keys["images_keys"],
                img_dict,
                additional_message=f" for image dict {img_dict}",
            )

        # Check keys in annotations dicts
        for annot_dict in data["annotations"]:
            _check_keys(
                self.required_keys["annotations_keys"],
                annot_dict,
                additional_message=f" for annotation dict {annot_dict}",
            )

        # Check keys in categories dicts
        for cat_dict in data["categories"]:
            _check_keys(
                self.required_keys["categories_keys"],
                cat_dict,
                additional_message=f" for category dict {cat_dict}",
            )

    @schema.validator
    def _schema_contains_required_keys(self, attribute, value):
        """Ensure that the schema includes the required keys."""
        missing_keys = []
        for key, subkeys in self.required_keys.items():
            if key == "main":
                main_dict = value.get("properties", {})
                for subkey in subkeys:
                    if subkey not in main_dict:
                        missing_keys.append(subkey)
            else:
                key_modif = key.replace("_keys", "")
                nested_dict = (
                    value.get("properties", {})
                    .get(key_modif, {})
                    .get("items", {})
                    .get("properties", {})
                )
                for subkey in subkeys:
                    if subkey not in nested_dict:
                        missing_keys.append(f"{key_modif}/{subkey}")

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
            f"found in {list(data_dict.keys())}{additional_message}."
        )
