"""Validators for supported annotation files."""

import json
from pathlib import Path

import jsonschema
import jsonschema.exceptions
import jsonschema.validators
from attrs import define, field, validators


def get_default_via_schema() -> dict:
    """Read a VIA schema file."""
    via_schema_path = (
        Path(__file__).parent / "json_schemas" / "via_schema.json"
    )
    with open(via_schema_path) as file:
        via_schema_dict = json.load(file)
    return via_schema_dict


def get_default_coco_schema() -> dict:
    """Read a COCO schema file."""
    coco_schema_path = (
        Path(__file__).parent / "json_schemas" / "coco_schema.json"
    )
    with open(coco_schema_path) as file:
        coco_schema_dict = json.load(file)
    return coco_schema_dict


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
