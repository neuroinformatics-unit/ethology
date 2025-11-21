"""Validators for detection datasets."""

from attrs import define, field

from ethology.validators.utils import ValidDataset


@define
class ValidBboxDetectionsDataset(ValidDataset):
    """Class for valid ``ethology`` bounding box detections datasets.

    This class validates that the input dataset:

    - is an xarray Dataset,
    - has ``image_id``, ``space``, ``id`` as dimensions,
    - has ``position``, ``shape`` and ``confidence`` as data variables,
    - ``position`` and ``shape`` span at least the dimensions ``image_id``,
      ``space`` and ``id``,
    - ``confidence`` spans at least the dimensions ``image_id`` and ``id``.


    Attributes
    ----------
    dataset : xarray.Dataset
        The xarray dataset to validate.
    required_dims : set
        The set of required dimension names: ``image_id``, ``space`` and
        ``id``.
    required_data_vars : dict[str, set]
        A dictionary mapping data variable names to their required minimum
        dimensions:

        - ``position`` maps to ``image_id``, ``space`` and ``id``,
        - ``shape`` maps to ``image_id``, ``space`` and ``id``,
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


@define
class ValidBboxDetectionsEnsembleDataset(ValidDataset):
    """Class for valid ``ethology`` bounding box ensembledetections datasets.

    This class validates that the input dataset:

    - is an xarray Dataset,
    - has ``image_id``, ``space``, ``id`` and ``model`` as dimensions,
    - has ``position``, ``shape`` and ``confidence`` as data variables,
    - ``position`` and ``shape`` span at least the dimensions ``image_id``,
      ``space``, ``id`` and ``model``,
    - ``confidence`` spans at least the dimensions ``image_id``, ``id``
      and ``model``.


    Attributes
    ----------
    dataset : xarray.Dataset
        The xarray dataset to validate.
    required_dims : set
        The set of required dimension names: ``image_id``, ``space``, ``id``
         and ``model``.
    required_data_vars : dict[str, set]
        A dictionary mapping data variable names to their required minimum
        dimensions:

        - ``position`` maps to ``image_id``, ``space``, ``id`` and ``model``,
        - ``shape`` maps to ``image_id``, ``space``, ``id`` and ``model``,
        - ``confidence`` maps to ``image_id``, ``id`` and ``model``.

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
    required_dims: set = field(
        default={"image_id", "space", "id", "model"},
        init=False,
    )
    required_data_vars: dict = field(
        default={
            "position": {"image_id", "space", "id", "model"},
            "shape": {"image_id", "space", "id", "model"},
            "confidence": {"image_id", "id", "model"},
        },
        init=False,
    )
