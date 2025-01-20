"""Utility functions for JSON schema files."""

import json
from pathlib import Path


def get_default_via_schema() -> dict:
    """Read a VIA schema file."""
    via_schema_path = Path(__file__).parent / "via_schema.json"
    with open(via_schema_path) as file:
        via_schema_dict = json.load(file)
    return via_schema_dict


def get_default_coco_schema() -> dict:
    """Read a COCO schema file."""
    coco_schema_path = Path(__file__).parent / "coco_schema.json"
    with open(coco_schema_path) as file:
        coco_schema_dict = json.load(file)
    return coco_schema_dict
