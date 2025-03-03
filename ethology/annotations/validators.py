"""Validators for supported annotation files."""

import json
from pathlib import Path

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
    path : Path | str
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

    path: Path = field(converter=Path)
    schema: dict = field(
        default=_get_default_schema("VIA"),
        init=False,
    )
    required_keys: dict = field(
        default={
            "main": ["_via_img_metadata", "_via_attributes"],
            "images": ["filename"],
            "regions": ["shape_attributes"],
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
    path : Path | str
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

    path: Path = field(converter=Path)
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

    @path.validator
    def _file_contains_unique_image_IDs(self, attribute, value):
        """Ensure that the COCO JSON file contains unique image IDs.

        When exporting to COCO format, the VIA tool attempts to extract the
        image ID from the image filename using ``parseInt``. As a result, if
        two or more images have the same number-based filename, the image IDs
        can be non-unique (i.e., more image filenames than image IDs). This is
        probably a bug in the VIA tool, but we need to check for this issue.
        """
        with open(value) as file:
            data = json.load(file)

        # Get number of elements in "images" list
        n_images = len(data["images"])

        # Get the image IDs
        unique_image_ids = set([img["id"] for img in data["images"]])

        # Check for duplicate image IDs
        if n_images != len(unique_image_ids):
            raise ValueError(
                "The image IDs in the input COCO file are not unique. "
                f"There are {n_images} image entries, but only "
                f"{len(unique_image_ids)} unique image IDs."
            )
