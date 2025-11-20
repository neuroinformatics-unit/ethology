"""Utils for validating `ethology` objects."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import wraps

import xarray as xr
from attrs import define, field


@define
class ValidDataset(ABC):
    """An abstract base class for valid ``ethology`` datasets.

    It checks that the input dataset has:

    - required dimensions
    - required data variables

    Subclasses must define ``required_dims`` and ``required_data_vars``
    attributes.

    Attributes
    ----------
    dataset : xarray.Dataset
        The xarray dataset to validate.
    required_dims : set
        Set of required dimension names (defined by subclasses).
    required_data_vars : set
        Set of required data variable names (defined by subclasses).

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

    # Subclasses should override these abstract properties
    @property
    @abstractmethod
    def required_dims(self) -> set:
        """Subclasses must provide a required_dims property."""
        pass

    @property
    @abstractmethod
    def required_data_vars(self) -> dict[str, set]:
        """Subclasses must provide a required_data_vars property."""
        pass

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
        for (
            data_var,
            required_dims_in_data_var,
        ) in self.required_data_vars.items():
            missing_dims = required_dims_in_data_var - set(
                value.data_vars[data_var].coords
            )
            if missing_dims:
                raise ValueError(
                    f"Missing required dimensions ({sorted(missing_dims)}) "
                    f"in data variable '{data_var}'."
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
