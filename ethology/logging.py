"""Logging utilities for the ethology package."""

import movement.utils.logging as movement_logging


def log_error(error, message: str, logger_name: str = "ethology"):
    """Log an error message and return the Exception.

    The function wraps the ``movement`` logger function of the same name, but
    uses the logger name "ethology" as the default.

    Parameters
    ----------
    error : Exception
        The error to log and return.
    message : str
        The error message.
    logger_name : str, optional
        The name of the logger to use. Defaults to "ethology".

    Returns
    -------
    Exception
        The error that was passed in.

    """
    return movement_logging.log_error(error, message, logger_name=logger_name)


def log_warning(message: str, logger_name: str = "ethology"):
    """Log a warning message.

    The function wraps the ``movement`` logger function of the same name, but
    uses the logger name "ethology" as the default.

    Parameters
    ----------
    message : str
        The warning message.
    logger_name : str, optional
        The name of the logger to use. Defaults to "ethology".

    """
    movement_logging.log_warning(
        "WARNING: " + message,
        logger_name=logger_name,
    )
