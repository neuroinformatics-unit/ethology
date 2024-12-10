"""Utility functions to work with annotations in JSON format."""

import json
from pathlib import Path


def read_json_file_as_dict(
    file_path: Path,
) -> dict:
    """Read JSON file as dict.

    Parameters
    ----------
    file_path : str
        Path to the JSON file

    Returns
    -------
    dict
        Dictionary with the JSON data

    """
    try:
        with open(file_path) as file:
            return json.load(file)
    except FileNotFoundError as not_found_error:
        msg = f"File not found: {file_path}"
        raise ValueError(msg) from not_found_error
    except json.JSONDecodeError as decode_error:
        msg = f"Error decoding JSON data from file: {file_path}"
        raise ValueError(msg) from decode_error


def read_via_json_file_as_dict(file_path: Path) -> dict:
    """Read VIA JSON file as dict.

    Parameters
    ----------
    file_path : str
        Path to the VIA JSON file

    Returns
    -------
    dict
        Dictionary with the JSON data

    """
    # Read data
    data = read_json_file_as_dict(file_path)

    # Check the expected keys are defined in the JSON file
    expected_keys = [
        "_via_settings",
        "_via_img_metadata",
        "_via_attributes",
        "_via_data_format_version",
        "_via_image_id_list",
    ]

    for ky in expected_keys:
        if ky not in data:
            raise ValueError(
                f"Expected key '{ky}' not found in file: {file_path}"
            )

    return data


# def read_via_json_file_as_xarray(file_path: Path):


#     via_dict = read_via_json_file_as_dict(file_path)
