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

    """

    path: Path = field(validator=validators.instance_of(Path))

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


@define
class ValidVIAUntrackedJSON:
    """Class for validating VIA JSON files for untracked data.

    The validator ensures that the file matches the expected schema.
    The schema validation only checks the type for each specified
    key if it exists. It does not check for the presence of the keys.


    Attributes
    ----------
    path : pathlib.Path
        Path to the JSON file.

    Raises
    ------
    ValueError
        If the JSON file does not match the expected schema.

    Notes
    -----
    https://json-schema.org/understanding-json-schema/

    """

    # TODO: add a check for the presence of the keys
    # that I use in loading the data

    path: Path = field(validator=validators.instance_of(Path))

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

        # should have been validated with ValidJSON
        # already so this should work fine
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

    @path.validator
    def _file_contains_required_keys(self, attribute, value):
        """Ensure that the JSON file contains the required keys."""
        required_keys = {
            "main": ["_via_img_metadata", "_via_image_id_list"],
            "image_keys": ["filename", "regions"],
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
                required_keys["image_keys"],
                img_dict,
                additional_error_message=f"for {img_str}",
            )
            # Check keys for each region
            for region in img_dict["regions"]:
                _check_keys(required_keys["region_keys"], region)

                # Check keys under shape_attributes
                _check_keys(
                    required_keys["shape_attributes_keys"],
                    region["shape_attributes"],
                    additional_error_message=f"for region under {img_str}",
                )


@define
class ValidCOCOUntrackedJSON:
    """Class for validating COCO JSON files for untracked data.

    The validator ensures that the file matches the expected schema.
    The schema validation only checks the type for each specified
    key if it exists. It does not check for the presence of the keys.

    Attributes
    ----------
    path : pathlib.Path
        Path to the JSON file.

    Raises
    ------
    ValueError
        If the JSON file does not match the expected schema.

    Notes
    -----
    https://json-schema.org/understanding-json-schema/

    """

    path: Path = field(validator=validators.instance_of(Path))

    # TODO: add a check for the presence of the keys
    # that I use in loading the data

    @path.validator
    def _file_macthes_COCO_JSON_schema(self, attribute, value):
        """Ensure that the JSON file matches the expected schema."""
        # Define schema for VIA JSON file for untracked
        # (aka manually labelled) data
        COCO_JSON_schema = {
            "type": "object",
            "properties": {
                "info": {"type": "object"},
                "licenses": {
                    "type": "array",
                },
                "images": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "file_name": {"type": "string"},
                            "id": {"type": "integer"},
                            "width": {"type": "integer"},
                            "height": {"type": "integer"},
                        },
                    },
                },
                "annotations": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},  # annotation global ID
                            "image_id": {"type": "integer"},
                            "bbox": {
                                "type": "array",
                                "items": {"type": "integer"},
                            },
                            "category_id": {"type": "integer"},
                            "area": {"type": "integer"},
                            "iscrowd": {"type": "integer"},
                        },
                    },
                },
                "categories": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "integer"},
                            "name": {"type": "string"},
                            "supercategory": {"type": "string"},
                        },
                    },
                },
            },
        }

        # should have been validated with ValidJSON
        # already so this should work fine
        with open(value) as file:
            data = json.load(file)

        # check against schema
        try:
            jsonschema.validate(instance=data, schema=COCO_JSON_schema)
        except jsonschema.exceptions.ValidationError as val_err:
            raise ValueError(
                "The JSON data does not match "
                f"the provided schema: {COCO_JSON_schema}"
            ) from val_err

    @path.validator
    def _file_contains_required_keys(self, attribute, value):
        """Ensure that the JSON file contains the required keys."""
        required_keys = {
            "main": ["images", "annotations", "categories"],
            "image_keys": [
                "id",
                "file_name",
            ],  # add height and width of image?
            "annotations_keys": ["id", "image_id", "bbox", "category_id"],
            "categories_keys": ["id", "name", "supercategory"],
        }

        # Read data as dict
        with open(value) as file:
            data = json.load(file)

        # Check first level keys
        _check_keys(required_keys["main"], data)

        # Check keys in images dicts
        for img_dict in data["images"]:
            _check_keys(
                required_keys["image_keys"],
                img_dict,
                additional_error_message=f"for image dict {img_dict}",
            )

        # Check keys in annotations dicts
        for annot_dict in data["annotations"]:
            _check_keys(
                required_keys["annotations_keys"],
                annot_dict,
                additional_error_message=f"for annotation dict {annot_dict}",
            )

        # Check keys in categories dicts
        for cat_dict in data["categories"]:
            _check_keys(
                required_keys["categories_keys"],
                cat_dict,
                additional_error_message=f"for category dict {cat_dict}",
            )


def _check_keys(
    list_required_keys: list[str],
    data_dict: dict,
    additional_error_message: str = "",
):
    missing_keys = set(list_required_keys) - data_dict.keys()
    if missing_keys:
        raise ValueError(
            f"Required key(s) {missing_keys} not "
            f"found in {list(data_dict.keys())} "
            + additional_error_message
            + "."
        )
