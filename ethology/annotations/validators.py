import json
from pathlib import Path

import jsonschema
import jsonschema.exceptions
from attrs import define, field, validators

# 


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
    # https://stackoverflow.com/questions/16222633/how-would-you-design-json-schema-for-an-arbitrary-key

    @path.validator
    def _file_macthes_VIA_JSON_schema(self, attribute, value):
        """Ensure that the JSON file matches the expected schema."""
        # should the schema be an attribute?
        VIA_JSON_schema = {
            "type": "object",
            "properties": {
                "_via_settings": {
                    "type": "object",
                    "properties": {
                        "ui": {"type": "object"},
                        "core": {"type": "object"},
                        "project": {"type": "object"},
                    },
                },
                "_via_img_metadata": {
                    "type": "object",
                    "additionalProperties": {  # ---- does this work?
                        "type": "object",
                        "properties": {
                            "filename": {"type": "string"},
                            "size": {"type": "integer"},
                            "regions": {
                                "type": "list",  # does this work?
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
                                        },
                                    },
                                },
                            },
                            "file_attributes": {"type": "object"},
                        },
                    },
                },
                "_via_attributes": {
                    "type": "dict",
                    "properties": {
                        "region": {"type": "dict"},
                        "file": {"type": "dict"},
                    },
                },
                "_via_data_format_version": {"type": "string"},
                "_via_image_id_list": {"type": "list"},
            },
        }

        # should have been validated with ValidVIAUntrackedJSON
        with open(value) as file:
            data = json.load(file)

        # check schema
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


@define
class ValidCOCOUntrackedJSON:
    pass
