"""Validators for detection datasets."""

from typing import ClassVar

from attrs import define

from ethology.validators.utils import ValidDataset


@define
class ValidBboxDetectionsDataset(ValidDataset):
    """Class for valid ``ethology`` bounding box detections datasets.

    This class validates that the input dataset:

    - is an xarray Dataset,
    - has ``image_id``, ``space``, ``id`` as dimensions,
    - has ``position``, ``shape``, ``category`` and ``confidence`` as data
      variables,
    - ``position`` and ``shape`` span at least the dimensions ``image_id``,
      ``space`` and ``id``,
    - ``category`` and ``confidence`` span at least the dimensions
      ``image_id`` and ``id``.


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
        - ``shape`` maps to ``image_id``, ``space`` and ``id``,
        - ``category`` maps to ``image_id`` and ``id``,
        - ``confidence`` maps to ``image_id`` and ``id``.

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
        "confidence": {"image_id", "id"},
    }
