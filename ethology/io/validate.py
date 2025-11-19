"""Utils for validating `ethology` objects."""

from collections.abc import Callable
from functools import wraps


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
