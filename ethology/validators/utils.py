"""Utils for validating `ethology` objects."""

from abc import ABC
from collections.abc import Callable
from functools import wraps
from typing import ClassVar

import xarray as xr
from attrs import define, field


@define
class ValidDataset(ABC):
    """An abstract base class for valid ``ethology`` datasets.

    This class validates that the input dataset:

    - is an xarray Dataset
    - contains all required dimensions
    - contains all required data variables
    - has the correct dimensions for each data variable

    Subclasses must define ``required_dims`` and ``required_data_vars``
    class attributes.

    Attributes
    ----------
    dataset : xarray.Dataset
        The xarray dataset to validate.
    required_dims : ClassVar[set[str]]
        A set of required dimension names. This class attribute must be
        defined by any subclass inheriting from this class.
    required_data_vars : ClassVar[dict[str, set]]
        A dictionary mapping data variable names to their required dimensions.
        This class attribute must be defined by any subclass inheriting from
        this class.

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

    dataset: xr.Dataset = field()

    # class variables
    required_dims: ClassVar[set]
    required_data_vars: ClassVar[dict[str, set]]

    def __init_subclass__(cls, **kwargs):
        """Verify that subclasses define required class variables."""
        super().__init_subclass__(**kwargs)

        if not hasattr(cls, "required_dims"):
            raise TypeError(
                f"{cls.__name__} must define 'required_dims' class variable"
            )
        if not hasattr(cls, "required_data_vars"):
            raise TypeError(
                f"{cls.__name__} must define 'required_data_vars' "
                "class variable"
            )

    # Validators
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
        missing_vars = self.required_data_vars.keys() - set(value.data_vars)
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

    @dataset.validator
    def _check_dimensions_per_data_variable(self, attribute, value):
        """Ensure the dataset has all required dimensions."""
        error_messages = []
        for data_var, dims_per_data_var in self.required_data_vars.items():
            missing_dims = dims_per_data_var - set(
                value.data_vars[data_var].coords
            )
            if missing_dims:
                error_messages.append(
                    f"data variable '{data_var}' is missing "
                    f"dimensions {sorted(missing_dims)}"
                )

        if error_messages:
            raise ValueError(
                "Some data variables are missing required dimensions:\n  - "
                + "\n  - ".join(error_messages)
            )


def _check_output(validator: type):
    """Return a decorator that validates the output of a function."""

    def decorator(function: Callable) -> Callable:
        @wraps(function)  # to preserve function metadata
        def wrapper(*args, **kwargs):
            result = function(*args, **kwargs)
            validator(result)
            return result

        return wrapper

    return decorator


def _check_input(validator: type, input_index: int = 0):
    """Return a decorator that validates a specific input of a function.

    By default, the first input is validated. If the input index is
    larger than the number of inputs, no validation is performed.
    """

    def decorator(function: Callable) -> Callable:
        @wraps(function)
        def wrapper(*args, **kwargs):
            if len(args) > input_index:
                validator(args[input_index])
            result = function(*args, **kwargs)
            return result

        return wrapper

    return decorator
