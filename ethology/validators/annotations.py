"""Validators for annotation files and datasets."""

import json
from pathlib import Path
from typing import ClassVar

import pandas as pd
import pandera.pandas as pa
from attrs import define, field
from pandera.typing import Index

from ethology.validators.json_schemas.utils import (
    _check_file_is_json,
    _check_file_matches_schema,
    _check_required_keys_in_dict,
    _get_default_schema,
)
from ethology.validators.utils import ValidDataset


@define
class ValidVIA:
    """Class for valid VIA JSON annotation files.

    It checks the input file is a valid JSON file, matches
    the VIA schema and contains the required keys.


    Attributes
    ----------
    path : Path | str
        Path to the VIA JSON file, passed as an input.
    schema : ClassVar[dict]
        The JSON schema is set to the default VIA schema.
    required_keys : ClassVar[dict]
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

    # class variables: should not be modified after initialization
    schema: ClassVar[dict] = _get_default_schema("VIA")
    required_keys: ClassVar[dict] = {
        "main": ["_via_img_metadata", "_via_attributes"],
        "images": ["filename"],
        "regions": ["shape_attributes"],
        "shape_attributes": ["x", "y", "width", "height"],
    }

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
    """Class for valid COCO JSON annotation files.

    It checks the input file is a valid JSON file, matches
    the COCO schema and contains the required keys.

    Attributes
    ----------
    path : Path | str
        Path to the COCO JSON file, passed as an input.
    schema : ClassVar[dict]
        The JSON schema is set to the default COCO schema.
    required_keys : ClassVar[dict]
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

    # class variables: should not be modified after initialization
    schema: ClassVar[dict] = _get_default_schema("COCO")
    required_keys: ClassVar[dict] = {
        "main": ["images", "annotations", "categories"],
        "images": ["id", "file_name", "width", "height"],
        "annotations": ["id", "image_id", "bbox", "category_id"],
        "categories": ["id", "name"],  # exclude "supercategory"
    }  # keys match the 1st level keys in a COCO JSON file

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


@define
class ValidBboxAnnotationsDataset(ValidDataset):
    """Class for valid ``ethology`` bounding box annotations datasets.

    This class validates that the input dataset:

    - is an xarray Dataset,
    - has ``image_id``, ``space``, ``id`` as dimensions,
    - has ``position``, ``shape`` and ``category`` as data variables,
    - ``position`` and ``shape`` span at least the dimensions ``image_id``,
      ``space`` and ``id``.
    - ``category`` spans at least the dimensions ``image_id`` and ``id``.


    Attributes
    ----------
    dataset : xarray.Dataset
        The xarray dataset to validate.
    required_dims : ClassVar[set]
        The set of required dimension names: ``image_id``, ``space`` and
        ``id``.
    required_data_vars : ClassVar[dict[str, set]]
        A dictionary mapping data variable names to their required minimum
        dimensions:

        - ``position`` maps to ``image_id``, ``space`` and ``id``,
        - ``shape`` maps to ``image_id``, ``space`` and ``id``.
        - ``category`` maps to ``image_id`` and ``id``.

    Raises
    ------
    TypeError
        If the input is not an xarray Dataset.
    ValueError
        If the dataset is missing required data variables or dimensions,
        or if any required dimensions are missing for any data variable.

    Notes
    -----
    The dataset can have other data variables and dimensions, but only the
    required ones are checked.

    """

    # Minimum requirements for a bbox dataset holding detections
    # Should not be modified after initialization
    required_dims: ClassVar[set] = {"image_id", "space", "id"}
    required_data_vars: ClassVar[dict[str, set]] = {
        "position": {"image_id", "space", "id"},
        "shape": {"image_id", "space", "id"},
        "category": {"image_id", "id"},
    }


class ValidBboxAnnotationsDataFrame(pa.DataFrameModel):
    """Class for valid bounding boxes intermediate dataframes.

    We use this dataframe internally as an intermediate step in the process of
    converting an input bounding box annotations file (VIA or COCO) to
    an ``ethology`` dataset. The validation checks all required columns
    exist and their types are correct.

    Attributes
    ----------
    image_filename : str
        Name of the image file.
    image_id : int
        Unique identifier for each of the images.
    image_width : int
        Width of each of the images, in the same units as the input file
        (usually pixels).
    image_height : int
        Height of each of the images, in the same units as the input file
        (usually pixels).
    x_min : float
        Minimum x-coordinate of the bounding box, in the same units as
        the input file.
    y_min : float
        Minimum y-coordinate of the bounding box, in the same units as
        the input file.
    width : float
        Width of the bounding box, in the same units as the input file.
    height : float
        Height of the bounding box, in the same units as the input file.
    category_id : int
        Unique identifier for the category, as specified in the input file.
        A value of 0 is usually reserved for the background class.
    category : str
        Category of the annotation as a string.
    supercategory : str
        Supercategory of the annotation as a string.

    Raises
    ------
    pa.errors.SchemaError
        If the input dataframe does not match the schema.

    See Also
    --------
    :class:`pandera.api.pandas.model.DataFrameModel`

    """

    # image columns
    image_filename: str = pa.Field(description="Name of the image file.")
    image_id: int = pa.Field(
        description="Unique identifier for each of the images."
    )
    image_width: int = pa.Field(
        description="Width of each of the images, "
        "in the same units as the input file (usually pixels)."
        # if not defined, it should be set to 0 in the df
    )
    image_height: int = pa.Field(
        description="Height of each of the images, "
        "in the same units as the input file (usually pixels)."
        # if not defined, it should be set to 0 in the df
    )

    # bbox columns
    x_min: float = pa.Field(
        description=(
            "Minimum x-coordinate of the bounding box, "
            "in the same units as the input file."
        )
    )
    y_min: float = pa.Field(
        description=(
            "Minimum y-coordinate of the bounding box, "
            "in the same units as the input file."
        )
    )
    width: float = pa.Field(
        description=(
            "Width of the bounding box, in the same units as the input file."
        )
    )
    height: float = pa.Field(
        description=(
            "Height of the bounding box, in the same units as the input file."
        )
    )

    # category columns
    # - always defined in COCO files exported with VIA tool
    # - optionally defined in VIA files exported with VIA tool
    category_id: int = pa.Field(
        description=(
            "Unique identifier for the category, "
            "as specified in the input file. A value of 0 "
            "is usually reserved for the background class."
        )
    )
    category: str = pa.Field(
        description="Category of the annotation as a string."
    )
    supercategory: str = pa.Field(
        description="Supercategory of the annotation as a string."
    )

    @staticmethod
    def get_empty_values() -> dict:
        """Get the default empty values for selected dataframe columns.

        The columns are those that can be undefined in VIA and COCO files:
        ``category``, ``supercategory``, ``category_id``, ``image_width`` and
        ``image_height``.

        Returns
        -------
        dict
            A dictionary with the default empty values the specified columns.

        """
        return {
            "category": "",  # it can be undefined in VIA files
            "supercategory": "",  # it can be undefined in VIA and COCO files
            "category_id": -1,  # it can be undefined in VIA files
            "image_width": 0,  # it can be undefined in VIA files
            "image_height": 0,  # it can be undefined in VIA files
        }


class ValidBboxAnnotationsCOCO(pa.DataFrameModel):
    """Class for COCO-exportable bounding box annotations dataframes.

    The validation checks the required columns exist and their types are
    correct. It additionally checks that the index and the
    ``annotation_id`` column are equal.

    Attributes
    ----------
    idx : Index[int]
        Index of the dataframe. Should be greater than or equal to 0 and equal
        to the ``annotation_id`` column.
    annotation_id : int
        Unique identifier for the annotation. Should be equal to the index.
    image_id : int
        Unique identifier for each of the images.
    image_filename : str
        Filename of the image.
    image_width : int
        Width of each of the images.
    image_height : int
        Height of each of the images.
    bbox : list[float]
        Bounding box coordinates as xmin, ymin, width, height.
    area : float
        Bounding box area.
    segmentation : list[list[float]]
        Bounding box segmentation masks as list of lists of coordinates.
    category : str
        Category of the annotation.
    supercategory : str
        Supercategory of the annotation.
    iscrowd : int
        Whether the annotation is a crowd. Should be 0 or 1.

    Raises
    ------
    pa.errors.SchemaError
        If the dataframe does not match the schema.

    Notes
    -----
    See `COCO format documentation <https://cocodataset.org/#format-data>`_
    for more details.

    See Also
    --------
    :class:`pandera.api.pandas.model.DataFrameModel`

    """

    # index
    idx: Index[int] = pa.Field(ge=0, check_name=False)

    # annotation_id
    annotation_id: int = pa.Field(
        description="Unique identifier for the annotation (index)",
    )

    # image columns
    image_id: int = pa.Field(
        description="Unique identifier for the image",
    )
    image_filename: str = pa.Field(
        description="Filename of the image",
    )
    image_width: int = pa.Field(
        description="Width of the image", ge=0, nullable=True
    )
    image_height: int = pa.Field(
        description="Height of the image", ge=0, nullable=True
    )

    # bbox data
    bbox: list[float] = pa.Field(
        description="Bounding box coordinates as xmin, ymin, width, height"
    )
    area: float = pa.Field(
        description="Bounding box area",
        ge=0,
    )
    segmentation: list[list[float]] = pa.Field(
        description="Bounding box segmentation as list of lists of coordinates"
    )

    # category columns
    # we do not require supercategories to be present in the
    # dataframe since they are not currently added to the xarray dataset
    category: str = pa.Field(
        description="Category of the annotation",
        nullable=True,
    )

    # other
    iscrowd: int = pa.Field(
        description="Whether the annotation is a crowd",
        isin=[0, 1],
        nullable=True,
    )

    @staticmethod
    def map_df_columns_to_COCO_fields() -> dict:
        """Map COCO-exportable dataframe columns to COCO fields.

        Returns
        -------
        dict
            A dictionary mapping each column in the COCO-exportable dataframe
            to the corresponding fields in the equivalent COCO file.

        """
        return {
            "images": {
                "image_id": "id",
                "image_filename": "file_name",
                "image_width": "width",
                "image_height": "height",
            },
            "categories": {
                "category_id": "id",
                "category": "name",
                "supercategory": "supercategory",
            },
            "annotations": {
                "annotation_id": "id",
                "area": "area",
                "bbox": "bbox",
                "image_id": "image_id",
                "category_id": "category_id",
                "iscrowd": "iscrowd",
                "segmentation": "segmentation",
            },
        }

    @pa.dataframe_check
    def check_idx_and_annotation_id(cls, df: pd.DataFrame) -> bool:
        """Check that the index and the ``annotation_id`` column are equal.

        Parameters
        ----------
        df : pd.DataFrame
            The dataframe to check.

        Returns
        -------
        bool
            A boolean indicating whether the index and the
            ``annotation_id`` column are equal for all rows.

        """
        return all(df.index == df["annotation_id"])
