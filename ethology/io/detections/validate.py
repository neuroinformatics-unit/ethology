"""Validators for detection datasets."""

import xarray as xr
from attrs import define, field


@define
class ValidBboxDetectionsDataset:
    """Class for valid ``ethology`` bounding box detections datasets.

    It checks that the input dataset has:

    - ``image_id``, ``space``, ``id`` as dimensions
    - ``position``, ``shape`` and ``confidence`` as data variables

    Attributes
    ----------
    dataset : xarray.Dataset
        The xarray dataset to validate.

    Raises
    ------
    TypeError
        If the input is not an xarray Dataset.
    ValueError
        If the dataset is missing required data variables or dimensions.

    Notes
    -----
    The dataset can have other data variables and dimensions, but only the
    required ones are checked.

    """

    dataset: xr.Dataset = field()

    # Minimum requirements for annotations datasets holding bboxes
    required_dims: set = field(
        default={"image_id", "space", "id"},
        init=False,
    )
    required_data_vars: set = field(
        default={"position", "shape", "confidence"},
        init=False,
    )

    @dataset.validator
    def _check_dataset_type(self, attribute, value):
        """Ensure the input is an xarray Dataset."""
        if not isinstance(value, xr.Dataset):
            raise TypeError(
                f"Expected an xarray Dataset, but got {type(value)}."
            )

    @dataset.validator
    def _check_required_data_variables(self, attribute, value):
        """Ensure the dataset has all required data variables."""
        missing_vars = self.required_data_vars - set(value.data_vars)
        if missing_vars:
            raise ValueError(
                f"Missing required data variables: {sorted(missing_vars)}"
            )

    @dataset.validator
    def _check_required_dimensions(self, attribute, value):
        """Ensure the dataset has all required dimensions."""
        missing_dims = self.required_dims - set(value.dims)
        if missing_dims:
            raise ValueError(
                f"Missing required dimensions: {sorted(missing_dims)}"
            )
