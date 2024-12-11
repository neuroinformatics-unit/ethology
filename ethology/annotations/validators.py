"""Validators for annotation files."""

import json
from pathlib import Path

import jsonschema
import jsonschema.exceptions
from attrs import define, field, validators


@define
class ValidJSON:
    """Class for validating JSON files.

    Attributes
    ----------
    path : pathlib.Path
        Path to the JSON file.

    Raises
    ------
    ValueError
        If the file is not in JSON format or if it does not contain the
        expected keys.

    """

    path: Path = field(validator=validators.instance_of(Path))

    @path.validator
    def _file_is_json(self, attribute, value):
        """Ensure that the file is a JSON file."""
        try:
            with open(value) as file:
                json.load(file)
        except FileNotFoundError as not_found_error:
            raise ValueError(f"File not found: {value}") from not_found_error
        except json.JSONDecodeError as decode_error:
            raise ValueError(
                f"Error decoding JSON data from file: {value}"
            ) from decode_error


@define
class ValidVIAUntrackedJSON:
    """Class for validating VIA JSON files for untracked data.

    The validator ensures that the file matches the expected schema.

    https://json-schema.org/understanding-json-schema/reference/object#additional-properties

    Attributes
    ----------
    path : pathlib.Path
        Path to the JSON file.

    Raises
    ------
    ValueError
        If the JSON file does not match the expected schema.

    """

    path: Path = field(validator=validators.instance_of(Path))
    # expected_schema: dict = field(factory=dict, kw_only=True)

    @path.validator
    def _file_macthes_VIA_JSON_schema(self, attribute, value):
        """Ensure that the JSON file matches the expected schema."""
        # Define schema for VIA JSON file for untracked
        # (aka manually labelled) data
        VIA_JSON_schema = {
            "type": "object",
            "properties": {
                # settings for browser UI
                "_via_settings": {
                    "type": "object",
                    "properties": {
                        "ui": {"type": "object"},
                        "core": {"type": "object"},
                        "project": {"type": "object"},
                    },
                },
                # annotation data
                "_via_img_metadata": {
                    "type": "object",
                    "additionalProperties": {
                        # "additionalProperties" to allow any key,
                        # see https://stackoverflow.com/a/69811612/24834957
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "size": {"type": "integer"},
                            "regions": {
                                "type": "array",  # a list of dicts
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "shape_attributes": {
                                            "type": "object",
                                            "properties": {
                                                "name": {"type": "string"},
                                                "x": {"type": "integer"},
                                                "y": {"type": "integer"},
                                                "width": {"type": "integer"},
                                                "height": {"type": "integer"},
                                            },
                                            "region_attributes": {
                                                "type": "object"
                                            },  # we just check it's a dict
                                        },
                                    },
                                },
                            },
                            "file_attributes": {"type": "object"},
                        },
                    },
                },
                # ordered list of image keys
                # - the position defines the image ID
                "_via_image_id_list": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                # region (aka annotation) and file attributes for VIA UI
                "_via_attributes": {
                    "type": "object",
                    "properties": {
                        "region": {"type": "object"},
                        "file": {"type": "object"},
                    },
                },
                # version of the VIA data format
                "_via_data_format_version": {"type": "string"},
            },
        }

        # should have been validated with ValidVIAUntrackedJSON
        with open(value) as file:
            data = json.load(file)

        # check against schema
        try:
            jsonschema.validate(instance=data, schema=VIA_JSON_schema)
        except jsonschema.exceptions.ValidationError as val_err:
            raise ValueError(
                "The JSON data does not match "
                f"the provided schema: {VIA_JSON_schema}"
            ) from val_err
        # except jsonschema.exceptions.SchemaError as schema_err:
        #     raise ValueError(
        #         f"Invalid schema provided: {VIA_JSON_schema}"
        #     ) from schema_err


# @define
# class ValidCOCOUntrackedJSON:
#     pass
