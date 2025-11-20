"""Validators for detection datasets."""

from attrs import define, field

from ethology.io.validate import ValidDataset


@define
class ValidBboxDetectionsDataset(ValidDataset):
    """Class for valid ``ethology`` bounding box detections datasets.

    It checks that the input dataset has:

    - ``image_id``, ``space``, ``id`` as dimensions
    - ``position``, ``shape`` and ``confidence`` as data variables

    Attributes
    ----------
    dataset : xarray.Dataset
        The xarray dataset to validate.
    required_dims : set
        Set of required dimension names.
    required_data_vars : dict[str, set]
        A dictionary mapping data variable names to their required dimensions.

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

    # Minimum requirements for a bbox dataset holding detections
    required_dims: set = field(
        default={"image_id", "space", "id"},
        init=False,
    )
    required_data_vars: dict = field(
        default={
            "position": {"image_id", "space", "id"},
            "shape": {"image_id", "space", "id"},
            "confidence": {"image_id", "id"},
        },
        init=False,
    )
