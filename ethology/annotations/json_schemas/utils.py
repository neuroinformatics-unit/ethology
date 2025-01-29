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
            f"Required key(s) {sorted(missing_keys)} not found " "in schema."
        )


def _check_required_keys_in_dict(
    list_required_keys: list[str],
    data: dict,
    additional_message: str = "",
):
    """Check if the required keys are present in the input dictionary."""
    missing_keys = set(list_required_keys) - set(data.keys())
    if missing_keys:
        raise ValueError(
            f"Required key(s) {sorted(missing_keys)} not "
            f"found{additional_message}."
        )


def _extract_properties_keys(schema: dict, parent_key="") -> list:
    """Recursively extract the keys of all "properties" subdictionaries.

    Recursively extract the keys of all subdictionaries in the input
    dictionary that are values to a "properties" key. The input dictionary
    represents a JSON schema dictionary
    (see https://json-schema.org/understanding-json-schema/about).

    The "properties" key always appears as part of a dictionary with at least
    another key, that is "type" or "item".
    """
    # The "property keys" are either "properties" or "additionalProperties"
    # as they are the keys with the relevant data
    property_keys = ["properties", "additionalProperties"]

    def _contains_properties_key(input: dict):
        """Return True if the input dictionary contains a property key."""
        return any(x in input for x in property_keys)

    def _get_properties_subdict(input: dict):
        """Get the subdictionary under the property key."""
        return input[next(k for k in property_keys if k in input)]

    keys_of_properties_dicts = []
    if "type" in schema:
        if _contains_properties_key(schema):
            # Get the subdictionary under the properties key
            properties_subdict = _get_properties_subdict(schema)

            # Check if there is a nested "properties" dict inside the current
            # one. If so, go down one level.
            if _contains_properties_key(properties_subdict):
                properties_subdict = _get_properties_subdict(
                    properties_subdict
                )

            # Add keys of deepest "properties dict" to list
            keys_of_properties_dicts.extend(
                [
                    f"{parent_key}/{ky}" if parent_key else ky
                    for ky in properties_subdict
                ]
            )

            # Inspect non-properties dictionaries under this properties subdict
            for ky, val in properties_subdict.items():
                full_key = f"{parent_key}/{ky}" if parent_key else ky
                keys_of_properties_dicts.extend(
                    _extract_properties_keys(val, full_key)
                )

        elif "items" in schema:
            # Analyse the dictionary under the "items" key
            properties_subdict = schema["items"]
            keys_of_properties_dicts.extend(
                _extract_properties_keys(
                    properties_subdict, parent_key=parent_key
                )
            )

    return sorted(keys_of_properties_dicts)
