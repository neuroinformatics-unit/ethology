from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from ethology.io.detections.validate import ValidBboxDetectionsDataset


@pytest.fixture
def valid_bbox_detections_dataset():
    """Create a valid bbox detections dataset for validation."""
    image_ids = [1, 2, 3]
    annotation_ids = [0, 1, 2]  # max 3 bboxes per frame
    space_dims = ["x", "y"]

    # Create position, shape and confidence data all zeros
    position_data = np.zeros(
        (len(image_ids), len(space_dims), len(annotation_ids))
    )
    shape_data = np.copy(position_data)
    confidence_data = np.zeros((len(image_ids), len(annotation_ids)))

    # Create the dataset
    ds = xr.Dataset(
        data_vars={
            "position": (["image_id", "space", "id"], position_data),
            "shape": (["image_id", "space", "id"], shape_data),
            "confidence": (["image_id", "id"], confidence_data),
        },
        coords={
            "image_id": image_ids,
            "space": ["x", "y"],
            "id": annotation_ids,
        },
    )

    return ds


@pytest.fixture
def valid_bbox_detections_dataset_extra_vars_and_dims(
    valid_bbox_detections_dataset: xr.Dataset,
) -> xr.Dataset:
    ds = valid_bbox_detections_dataset.copy(deep=True)
    ds.coords["extra_dim"] = [10, 20, 30]
    ds["extra_var_1"] = (["image_id"], np.random.rand(len(ds.image_id)))
    ds["extra_var_2"] = (["id"], np.random.rand(len(ds.id)))
    return ds


@pytest.mark.parametrize(
    "sample_dataset, expected_exception, expected_error_message",
    [
        (
            "valid_bbox_detections_dataset",
            does_not_raise(),
            "",
        ),
        (
            "valid_bbox_detections_dataset_extra_vars_and_dims",
            does_not_raise(),
            "",
        ),
        (
            xr.Dataset(
                coords={
                    "image_id": np.arange(3),
                    "space": np.arange(2),
                    "id": np.arange(2),
                },
                data_vars={
                    "position": (
                        ["image_id", "space", "id"],
                        np.zeros((3, 2, 2)),
                    ),
                    "shape": (
                        ["image_id", "space", "id", "foo"],
                        np.zeros((3, 2, 2, 1)),
                    ),
                    "confidence": (
                        ["image_id", "id"],
                        np.zeros((3, 2)),
                    ),
                },
            ),
            does_not_raise(),
            "",
        ),
        (
            {"position": [1, 2, 3], "shape": [4, 5, 6]},
            pytest.raises(TypeError),
            "Expected an xarray Dataset, but got <class 'dict'>.",
        ),
        (
            xr.Dataset(
                coords={
                    "image_id": np.arange(3),
                    "space": ["x", "y"],
                    "id": np.arange(2),
                },
                data_vars={
                    "position": (
                        ["image_id", "space", "id"],
                        np.zeros((3, 2, 2)),
                    ),
                    "shape": (
                        ["image_id", "space", "id"],
                        np.zeros((3, 2, 2)),
                    ),
                },
            ),
            pytest.raises(ValueError),
            "Missing required data variables: ['confidence']",
        ),
        (
            xr.Dataset(
                coords={
                    "image_id": np.arange(3),
                    "space": ["x", "y"],
                    "id": np.arange(2),
                },
                data_vars={
                    "position": (
                        ["image_id", "space", "id"],
                        np.zeros((3, 2, 2)),
                    ),
                },
            ),
            pytest.raises(ValueError),
            "Missing required data variables: ['confidence', 'shape']",
        ),
        (
            xr.Dataset(
                coords={"image_id": np.arange(3), "id": np.arange(2)},
                data_vars={
                    "position": (["image_id", "id"], np.zeros((3, 2))),
                    "shape": (["image_id", "id"], np.zeros((3, 2))),
                    "confidence": (["image_id", "id"], np.zeros((3, 2))),
                },
            ),
            pytest.raises(ValueError),
            "Missing required dimensions: ['space']",
        ),
        (
            xr.Dataset(
                coords={
                    "foo": np.arange(3),
                    "bar": ["x", "y"],
                    "id": np.arange(2),
                },
                data_vars={
                    "position": (
                        ["foo", "bar", "id"],
                        np.zeros((3, 2, 2)),
                    ),
                    "shape": (
                        ["foo", "bar", "id"],
                        np.zeros((3, 2, 2)),
                    ),
                    "confidence": (
                        ["foo", "id"],
                        np.zeros((3, 2)),
                    ),
                },
            ),
            pytest.raises(ValueError),
            "Missing required dimensions: ['image_id', 'space']",
        ),
        (
            xr.Dataset(
                coords={
                    "image_id": np.arange(3),
                    "space": np.arange(2),
                    "id": np.arange(2),
                },
                data_vars={
                    "position": (
                        ["image_id", "space", "id"],
                        np.zeros((3, 2, 2)),
                    ),
                    "shape": (
                        ["image_id", "id"],
                        np.zeros((3, 2)),
                    ),
                    "confidence": (
                        ["image_id", "id"],
                        np.zeros((3, 2)),
                    ),
                },
            ),
            pytest.raises(ValueError),
            (
                "Some data variables are missing required dimensions:"
                "\n  - data variable 'shape' is missing dimensions ['space']"
            ),
        ),
    ],
    ids=[
        "valid_bbox_detections",
        "valid_bbox_detections_extra_vars_and_dims",
        "valid_bbox_detections_extra_dims_in_shape_var",
        "invalid_bbox_detections_type",
        "invalid_bbox_detections_dataset_missing_data_var",
        "invalid_bbox_detections_missing_multiple_data_vars",
        "invalid_bbox_detections_missing_dimension",
        "invalid_bbox_detections_missing_multiple_dimensions",
        "invalid_bbox_detections_missing_dimension_in_data_var",
    ],
)
def test_validator_bbox_detections_dataset(
    sample_dataset: str | dict,
    expected_exception: pytest.raises,
    expected_error_message: str,
    request: pytest.FixtureRequest,
):
    """Test bbox annotations dataset validation in various input scenarios."""
    # Get dataset to validate
    if isinstance(sample_dataset, str):
        dataset = request.getfixturevalue(sample_dataset)
    else:
        dataset = sample_dataset

    # Run validation and check exception
    with expected_exception as excinfo:
        validator = ValidBboxDetectionsDataset(dataset=dataset)

    if excinfo:
        error_msg = str(excinfo.value)
        assert error_msg in expected_error_message
    else:
        assert validator.dataset is dataset
        assert validator.required_dims == {"image_id", "space", "id"}
        assert validator.required_data_vars == {
            "position": {"image_id", "space", "id"},
            "shape": {"image_id", "space", "id"},
            "confidence": {"image_id", "id"},
        }
