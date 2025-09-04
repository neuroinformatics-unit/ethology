"""Pytest fixtures shared across annotations tests."""

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pooch
import pytest
import xarray as xr


@pytest.fixture()
def annotations_test_data(
    pooch_registry: pooch.Pooch, get_paths_test_data: Callable
) -> dict:
    """Return the paths of the test files under the annotations subdirectory
    in the GIN test data repository.
    """
    return get_paths_test_data(pooch_registry, "test_annotations")


# ----------------- File validator fixtures -----------------


@pytest.fixture()
def json_file_decode_error(tmp_path: Path) -> Path:
    """Return path to a JSON file with a decoding error."""
    json_file = tmp_path / "JSON_decode_error.json"
    with open(json_file, "w") as f:
        f.write("just-a-string")
    return json_file


@pytest.fixture()
def json_file_not_found_error(tmp_path: Path) -> Path:
    """Return path to a JSON file that does not exist."""
    return tmp_path / "JSON_file_not_found.json"


@pytest.fixture()
def VIA_file_schema_mismatch(
    annotations_test_data: dict,
    tmp_path: Path,
) -> Path:
    """Return path to a VIA JSON file that does not match its schema.

    Specifically, we modify the type of the "width" of the first bounding box
    in the first image, from "int" to "str"
    """
    # Read valid JSON file
    valid_VIA_file_sample_1 = annotations_test_data["VIA_JSON_sample_1.json"]
    with open(valid_VIA_file_sample_1) as f:
        data = json.load(f)

    # Modify file so that it doesn't match the corresponding schema
    # (make width a string)
    _, img_dict = list(data["_via_img_metadata"].items())[0]
    img_dict["regions"][0]["shape_attributes"]["width"] = "49"

    # Save the modified JSON to a new file
    out_json = tmp_path / f"{valid_VIA_file_sample_1.stem}_schema_error.json"
    with open(out_json, "w") as f:
        json.dump(data, f)
    return out_json


@pytest.fixture()
def COCO_file_schema_mismatch(
    annotations_test_data: dict,
    tmp_path: Path,
) -> Path:
    """Return path to a COCO JSON file that doesn't match its schema.

    Specifically, we modify the type of the object under the "annotations"
    key from "list of dicts" to "list"
    """
    # Read valid JSON file
    valid_COCO_file_sample_1 = annotations_test_data["COCO_JSON_sample_1.json"]
    with open(valid_COCO_file_sample_1) as f:
        data = json.load(f)

    # Modify file so that it doesn't match the corresponding schema
    data["annotations"] = [1, 2, 3]  # [d] for d in data["annotations"]]

    # save the modified json to a new file
    out_json = tmp_path / f"{valid_COCO_file_sample_1.stem}_schema_error.json"
    with open(out_json, "w") as f:
        json.dump(data, f)
    return out_json


@pytest.fixture()
def small_schema() -> dict:
    """Small schema with properties keys:
    ["a", "b", "b/b1", "c", "c/c1", "c/c2"].
    """
    return {
        "type": "object",
        "properties": {
            "a": {
                "type": "array",
                "items": {"type": "string"},
            },
            "b": {
                "type": "object",
                "properties": {"b1": {"type": "string"}},
            },
            "c": {
                "type": "object",
                "properties": {
                    "c1": {"type": "string"},
                    "c2": {"type": "string"},
                },
            },
        },
    }


@pytest.fixture()
def default_VIA_schema() -> dict:
    """Get default VIA schema."""
    from ethology.io.annotations.json_schemas.utils import _get_default_schema

    return _get_default_schema("VIA")


@pytest.fixture()
def default_COCO_schema() -> dict:
    """Get default COCO schema."""
    from ethology.io.annotations.json_schemas.utils import _get_default_schema

    return _get_default_schema("COCO")


# ----------------- Bboxes dataset validation fixtures -----------------
@pytest.fixture
def valid_bboxes_dataset():
    """Create a valid xarray dataset for bboxes validation."""
    image_ids = [1, 2, 3]
    annotation_ids = [0, 1, 2]  # three per frame
    space_dims = ["x", "y"]

    # Create position and shape data all zeros
    position_data = np.zeros(
        (len(image_ids), len(space_dims), len(annotation_ids))
    )
    shape_data = np.zeros((len(image_ids), len(annotation_ids)))

    # Create the dataset
    ds = xr.Dataset(
        data_vars={
            "position": (["image_id", "space", "id"], position_data),
            "shape": (["image_id", "id"], shape_data),
        },
        coords={
            "image_id": image_ids,
            "space": ["x", "y"],
            "id": annotation_ids,
        },
    )

    return ds


@pytest.fixture
def valid_bboxes_dataset_extra_vars_and_dims(
    valid_bboxes_dataset: xr.Dataset,
) -> xr.Dataset:
    ds = valid_bboxes_dataset.copy(deep=True)
    ds.coords["extra_dim"] = [10, 20, 30]
    ds["extra_var_1"] = (["image_id"], np.random.rand(len(ds.image_id)))
    ds["extra_var_2"] = (["id"], np.random.rand(len(ds.id)))
    return ds
