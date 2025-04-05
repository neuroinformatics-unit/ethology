"""Utility functions for JSON schema files."""

import json
from pathlib import Path

import jsonschema


def _get_default_schema(schema_name: str) -> dict:
    """Get the default VIA or COCO schema as a dictionary."""
    schema_path = (
        Path(__file__).parent / "schemas" / f"{schema_name}_schema.json"
    )
    with open(schema_path) as file:
        schema_dict = json.load(file)
    return schema_dict


def _check_file_is_json(filepath: Path):
    """Check the input file can be read as a JSON."""
    try:
        with open(filepath) as file:
            json.load(file)
    except json.JSONDecodeError as decode_error:
        # We override the error message for clarity
        raise ValueError(
            f"Error decoding JSON data from file: {filepath}. "
            "The data being deserialized is not a valid JSON. "
        ) from decode_error


def _check_file_matches_schema(filepath: Path, schema: dict | None):
    """Check the input JSON file matches the given schema.

    The schema validation only checks the type for each specified
    key if the key exists. It does not check that the keys in the
    schema are present in the JSON file.
    """
    # Read json file
    with open(filepath) as file:
        data = json.load(file)

    # Check against schema if provided
    if schema:
        jsonschema.validate(instance=data, schema=schema)


def _check_required_properties_keys(
    required_properties_keys: list, schema: dict
):
    """Check the input schema includes the required "properties" keys."""
    # Get keys of "properties" dictionaries in schema
    properties_keys_in_schema = _extract_properties_keys(schema)

    # Get list of "properties" keys that are required but not in schema
    missing_keys = set(required_properties_keys) - set(
        properties_keys_in_schema
    )

    # Raise error if there are missing keys in the schema
    if missing_keys:
        raise ValueError(
            f"Required key(s) {sorted(missing_keys)} not found in schema."
        )


def _check_required_keys_in_dict(
    list_required_keys: list[str],
    data: dict,
    additional_message: str = "",
):
    """Check if the required keys are present in the input dictionary.

    It checks that the required keys are defined in the input dictionary.
    If they are defined, and they are dictionaries or lists,
    it additionally checks that they are not empty.

    The additional_message parameter is used to provide additional context in
    the error message for the missing keys check.
    """
    missing_keys = set(list_required_keys) - set(data.keys())
    if missing_keys:
        raise ValueError(
            f"Required key(s) {sorted(missing_keys)} not "
            f"found{additional_message}."
        )
    else:
        keys_with_empty_values = [
            key
            for key in list_required_keys
            if isinstance(data[key], dict | list) and not data[key]
        ]
        if keys_with_empty_values:
            raise ValueError(
                f"Empty value(s) found for the required key(s) "
                f"{sorted(keys_with_empty_values)}."
            )


def _extract_properties_keys(input_schema: dict, prefix: str = "") -> list:
    """Extract keys from all "properties" subdictionaries in a JSON schema.

    Recursively extract the keys of all subdictionaries in the input
    dictionary that are values to a "properties" key. The input dictionary
    represents a JSON schema dictionary
    (see https://json-schema.org/understanding-json-schema/about). The output
    is a sorted list of strings with full paths (e.g. 'parent/child').

    The "properties" key always appears as part of a set of dictionary keys
    with at least another key being "type" or "item". We use this to find the
    relevant subdictionaries.

    """
    result: list[str] = []

    # Skip if "type" key is missing in the schema
    if "type" not in input_schema:
        return result

    # If the input dictionary has a "properties" key: extract keys
    # and recurse into nested dictionaries
    if "properties" in input_schema:
        for key, value in input_schema["properties"].items():
            full_key = f"{prefix}/{key}" if prefix else key
            result.append(full_key)
            # Recurse into nested dictionaries to look for more "properties"
            # dicts
            result.extend(_extract_properties_keys(value, full_key))

    # If dictionary has "additionalProperties" key: recurse into it
    if "additionalProperties" in input_schema:
        result.extend(
            _extract_properties_keys(
                input_schema["additionalProperties"],
                prefix,
            )
        )

    # If dictionary has "items" key: recurse into it
    if "items" in input_schema:
        result.extend(
            _extract_properties_keys(
                input_schema["items"],
                prefix,
            )
        )

    # Return sorted list of keys with full paths
    return sorted(result)
