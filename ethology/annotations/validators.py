"""Validators for supported annotation files."""

import json
from pathlib import Path

import jsonschema
import jsonschema.exceptions
import jsonschema.validators
from attrs import define, field, validators


def get_default_VIA_schema() -> dict:
    """Read the default VIA schema."""
    via_schema_path = (
        Path(__file__).parent / "json_schemas" / "VIA_schema.json"
    )
    with open(via_schema_path) as file:
        via_schema_dict = json.load(file)
    return via_schema_dict


def get_default_COCO_schema() -> dict:
    """Read the default COCO schema."""
    coco_schema_path = (
        Path(__file__).parent / "json_schemas" / "COCO_schema.json"
    )
    with open(coco_schema_path) as file:
        coco_schema_dict = json.load(file)
    return coco_schema_dict


@define
class ValidJSON:
    """Class for valid JSON files.

    Upon instantiation, it checks the JSON file exists and can be decoded,
    and optionally validates the file against a provided JSON schema.

    Attributes
    ----------
    path : pathlib.Path
        Path to the JSON file.

    schema : dict, optional
        JSON schema to validate the file against.

    Raises
    ------
    FileNotFoundError
        If the JSON file does not exist.
    ValueError
        If the JSON file cannot be decoded.
    jsonschema.exceptions.ValidationError
        If the type of the keys present in the JSON file
        does not match the corresponding type specified in
        the schema.
    jsonschema.exceptions.SchemaError
        If the schema is invalid, according to the specified
        meta-schema. If no meta-schema is defined in the input
        schema, the latest released draft of the JSON schema
        specification is used.

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
        """Ensure that the JSON file matches the provided schema.

        The schema validation only checks the type of a key if it exists in
        the input data. It does not check that all the keys in the
        schema are present in the JSON file.
        """
        if self.schema:
            # Read json file
            with open(value) as file:
                data = json.load(file)

            # Check against schema
            try:
                jsonschema.validate(instance=data, schema=self.schema)
            except jsonschema.exceptions.ValidationError as val_err:
                raise val_err
            except jsonschema.exceptions.SchemaError as schema_err:
                raise schema_err
