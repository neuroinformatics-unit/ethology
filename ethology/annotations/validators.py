"""Validators for supported annotation files."""

import ast
import json
from pathlib import Path

import pandas as pd
from attrs import define, field

from ethology.annotations.json_schemas.utils import (
    _check_file_is_json,
    _check_file_matches_schema,
    _check_required_keys_in_dict,
    _get_default_schema,
)


@define
class ValidVIA:
    """Class for valid VIA JSON files.

    It checks the input file is a valid JSON file, matches
    the VIA schema and contains the required keys.


    Attributes
    ----------
    path : pathlib.Path
        Path to the VIA JSON file, passed as an input.
    schema : dict
        The JSON schema is set to the default VIA schema.
    required_keys : dict
        The required keys for the VIA JSON file.

    Raises
    ------
    ValueError
        If the JSON file cannot be decoded.
    jsonschema.exceptions.ValidationError
        If the type of any of the keys in the JSON file
        does not match the type specified in the schema.
    jsonschema.exceptions.SchemaError
        If the schema is invalid.
    ValueError
        If the VIA JSON file is missing any of the required keys.

    """

    path: Path = field()
    schema: dict = field(
        default=_get_default_schema("VIA"),
        init=False,
    )
    required_keys: dict = field(
        default={
            "main": [
                "_via_img_metadata",
                "_via_image_id_list",
                "_via_attributes",
            ],
            "images": ["filename", "regions"],
            "regions": ["shape_attributes", "region_attributes"],
            "shape_attributes": ["x", "y", "width", "height"],
        },
        init=False,
        # with init=False the attribute is always initialized
        # with the default value
    )

    # Note: the validators are applied in order
    @path.validator
    def _file_is_json(self, attribute, value):
        _check_file_is_json(value)

    @path.validator
    def _file_matches_JSON_schema(self, attribute, value):
        _check_file_matches_schema(value, self.schema)

    @path.validator
    def _file_contains_required_keys(self, attribute, value):
        """Ensure that the VIA JSON file contains the required keys."""
        # Read data as dict
        with open(value) as file:
            data = json.load(file)

        # Check first level keys
        _check_required_keys_in_dict(self.required_keys["main"], data)

        # Check keys in nested dicts
        for img_str, img_dict in data["_via_img_metadata"].items():
            # Check keys for each image dictionary
            _check_required_keys_in_dict(
                self.required_keys["images"],
                img_dict,
                additional_message=f" for {img_str}",
            )

            # Check keys for each region in an image
            for i, region in enumerate(img_dict["regions"]):
                # Check keys under first level per region
                _check_required_keys_in_dict(
                    self.required_keys["regions"],
                    region,
                    additional_message=f" for region {i} under {img_str}",
                )

                # Check keys under "shape_attributes" per region
                _check_required_keys_in_dict(
                    self.required_keys["shape_attributes"],
                    region["shape_attributes"],
                    additional_message=f" for region {i} under {img_str}",
                )


@define
class ValidCOCO:
    """Class for valid COCO JSON files.

    It checks the input file is a valid JSON file, matches
    the COCO schema and contains the required keys.

    Attributes
    ----------
    path : pathlib.Path
        Path to the COCO JSON file, passed as an input.
    schema : dict
        The JSON schema is set to the default COCO schema.
    required_keys : dict
        The required keys for the COCO JSON file.

    Raises
    ------
    ValueError
        If the JSON file cannot be decoded.
    jsonschema.exceptions.ValidationError
        If the type of any of the keys in the JSON file
        does not match the type specified in the schema.
    jsonschema.exceptions.SchemaError
        If the schema is invalid.
    ValueError
        If the COCO JSON file is missing any of the required keys.

    """

    path: Path = field()
    schema: dict = field(
        default=_get_default_schema("COCO"),
        init=False,
        # with init=False the attribute is always initialized
        # with the default value
    )

    # The keys of "required_keys" match the 1st level keys in a COCO JSON file
    required_keys: dict = field(
        default={
            "main": ["images", "annotations", "categories"],
            "images": ["id", "file_name"],
            "annotations": ["id", "image_id", "bbox", "category_id"],
            "categories": ["id", "name", "supercategory"],
        },
        init=False,
    )

    # Note: the validators are applied in order
    @path.validator
    def _file_is_json(self, attribute, value):
        _check_file_is_json(value)

    @path.validator
    def _file_matches_JSON_schema(self, attribute, value):
        _check_file_matches_schema(value, self.schema)

    @path.validator
    def _file_contains_required_keys(self, attribute, value):
        """Ensure that the COCO JSON file contains the required keys."""

        # Helper function to singularise the input key for the
        # error message
        def _singularise_err_msg(key):
            return key[:-1] if key != "categories" else key[:-3] + "y"

        # Read file as dict
        with open(value) as file:
            data = json.load(file)

        # Check first level keys
        _check_required_keys_in_dict(self.required_keys["main"], data)

        # Check keys in every dict listed under the "images", "annotations"
        # and "categories" keys
        for ky in list(self.required_keys.keys())[1:]:
            for instance_dict in data[ky]:
                _check_required_keys_in_dict(
                    self.required_keys[ky],
                    instance_dict,
                    additional_message=(
                        f" for {_singularise_err_msg(ky)} {instance_dict}"
                    ),
                )


@define
class ValidVIAcsv:
    """Class for valid VIA CSV files.

    It checks the input CSV file contains the expected header and
    represents rectangular bounding boxes.

    Attributes
    ----------
    path : pathlib.Path
        Path to the VIA CSV file, passed as an input.
    required_keys : dict
        The required keys for the VIA CSV file.

    Raises
    ------
    ValueError
        If the VIA CSV file is missing any of the required keys.

    """

    path: Path = field()

    @path.validator
    def _check_file_contains_valid_header(self, attribute, value):
        """Ensure the VIA .csv file contains the expected header."""
        expected_header = [
            "filename",
            "file_size",
            "file_attributes",
            "region_count",
            "region_id",
            "region_shape_attributes",
            "region_attributes",
        ]

        with open(value) as f:
            header = f.readline().strip("\n").split(",")
            if header != expected_header:
                raise ValueError(
                    ".csv header row does not match the known format for "
                    "VIA .csv files. "
                    f"Expected {expected_header} but got {header}.",
                )

    @path.validator
    def _check_region_shape(self, attribute, value):
        df = pd.read_csv(value, sep=",", header=0)

        for row in df.itertuples():
            region_shape_attrs = ast.literal_eval(row.region_shape_attributes)

            # check annotation is a rectangle
            if region_shape_attrs["name"] != "rect":
                raise ValueError(
                    f"{row.filename} (row {row.Index}): "
                    "bounding box shape must be 'rect' (rectangular) "
                    "but instead got "
                    f"'{region_shape_attrs['name']}'.",
                )
