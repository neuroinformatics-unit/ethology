from contextlib import nullcontext as does_not_raise

import numpy as np
import pytest
import xarray as xr

from ethology.validators.detections import (
    ValidBboxDetectionsDataset,
    ValidBboxDetectionsEnsembleDataset,
)


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
def valid_bbox_detections_ensemble_dataset(valid_bbox_detections_dataset):
    """Create a valid bbox detections ensemble_dataset for validation."""
    # Add model dimension
    ds = valid_bbox_detections_dataset.expand_dims(
        model=["model_a", "model_b"]
    )

    return ds


@pytest.fixture
def valid_bbox_detections_ensemble_dataset_extra_vars_and_dims(
    valid_bbox_detections_ensemble_dataset: xr.Dataset,
) -> xr.Dataset:
    ds = valid_bbox_detections_ensemble_dataset.copy(deep=True)
    ds.coords["extra_dim"] = [10, 20, 30]
    ds["extra_var_1"] = (["image_id"], np.random.rand(len(ds.image_id)))
    ds["extra_var_2"] = (["id"], np.random.rand(len(ds.id)))
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


# Define validator configurations
VALIDATOR_CONFIGS: dict = {
    "detections_ds": {
        "validator_class": ValidBboxDetectionsDataset,
        "valid_fixture": "valid_bbox_detections_dataset",
        "valid_fixture_extra": (
            "valid_bbox_detections_dataset_extra_vars_and_dims"
        ),
        "required_dims": {"image_id", "space", "id"},
        "required_data_vars": {
            "position": {"image_id", "space", "id"},
            "shape": {"image_id", "space", "id"},
            "confidence": {"image_id", "id"},
        },
    },
    "ensemble_ds": {
        "validator_class": ValidBboxDetectionsEnsembleDataset,
        "valid_fixture": "valid_bbox_detections_ensemble_dataset",
        "valid_fixture_extra": (
            "valid_bbox_detections_ensemble_dataset_extra_vars_and_dims"
        ),
        "required_dims": {"image_id", "space", "id", "model"},
        "required_data_vars": {
            "position": {"image_id", "space", "id", "model"},
            "shape": {"image_id", "space", "id", "model"},
            "confidence": {"image_id", "id", "model"},
        },
    },
}


@pytest.mark.parametrize("validator_type", ["detections_ds", "ensemble_ds"])
@pytest.mark.parametrize(
    "valid_fixture_key",
    [
        "valid_fixture",
        "valid_fixture_extra",
    ],
)
def test_validator_bbox_detections_dataset_valid(
    validator_type: str,
    valid_fixture_key: str,
    request: pytest.FixtureRequest,
):
    """Test bbox detections dataset validation with valid datasets."""
    config = VALIDATOR_CONFIGS[validator_type]
    fixture_name = config[valid_fixture_key]
    dataset = request.getfixturevalue(fixture_name)

    validator_class = config["validator_class"]
    with does_not_raise():
        validator = validator_class(dataset=dataset)

    assert validator.dataset is dataset
    assert validator.required_dims == config["required_dims"]
    assert validator.required_data_vars == config["required_data_vars"]


@pytest.mark.parametrize(
    "validator",
    [ValidBboxDetectionsDataset, ValidBboxDetectionsEnsembleDataset],
)
@pytest.mark.parametrize(
    "sample_dataset, expected_exception, expected_error_message",
    [
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
    ],
    ids=[
        "invalid_type",
        "invalid_missing_data_var",
        "invalid_missing_multiple_data_vars",
    ],
)
def test_validator_bbox_detections_dataset_invalid(
    validator: type[ValidBboxDetectionsDataset]
    | type[ValidBboxDetectionsEnsembleDataset],
    sample_dataset: xr.Dataset,
    expected_exception: pytest.raises,
    expected_error_message: str,
):
    """Test bbox annotations dataset validation in various input scenarios."""
    # Run validation and check exception
    with expected_exception as excinfo:
        _validator = validator(dataset=sample_dataset)
    if excinfo:
        error_msg = str(excinfo.value)
        assert error_msg in expected_error_message


@pytest.mark.parametrize(
    "validator",
    [ValidBboxDetectionsDataset, ValidBboxDetectionsEnsembleDataset],
)
@pytest.mark.parametrize(
    "sample_dataset, expected_exception, expected_error_message",
    [
        (
            xr.Dataset(
                coords={
                    "image_id": np.arange(3),
                    "id": np.arange(2),
                    "model": np.arange(2),
                },
                data_vars={
                    "position": (
                        ["image_id", "id", "model"],
                        np.zeros((3, 2, 2)),
                    ),
                    "shape": (
                        ["image_id", "id", "model"],
                        np.zeros((3, 2, 2)),
                    ),
                    "confidence": (
                        ["image_id", "id", "model"],
                        np.zeros((3, 2, 2)),
                    ),
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
                    "model": np.arange(2),
                },
                data_vars={
                    "position": (
                        ["foo", "bar", "id", "model"],
                        np.zeros((3, 2, 2, 2)),
                    ),
                    "shape": (
                        ["foo", "bar", "id", "model"],
                        np.zeros((3, 2, 2, 2)),
                    ),
                    "confidence": (
                        ["foo", "id", "model"],
                        np.zeros((3, 2, 2)),
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
                    "model": np.arange(2),
                },
                data_vars={
                    "position": (
                        ["image_id", "space", "id", "model"],
                        np.zeros((3, 2, 2, 2)),
                    ),
                    "shape": (
                        ["image_id", "id", "model"],
                        np.zeros((3, 2, 2)),
                    ),
                    "confidence": (
                        ["image_id", "id", "model"],
                        np.zeros((3, 2, 2)),
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
        "invalid_missing_dimension",
        "invalid_missing_multiple_dimensions",
        "invalid_missing_dimension_in_data_var",
    ],
)
def test_validator_bbox_detections_dataset_missing_dims(
    validator: type[ValidBboxDetectionsDataset]
    | type[ValidBboxDetectionsEnsembleDataset],
    sample_dataset: xr.Dataset,
    expected_exception: pytest.raises,
    expected_error_message: str,
):
    # Run validation and check exception
    with expected_exception as excinfo:
        _validator = validator(dataset=sample_dataset)
    if excinfo:
        error_msg = str(excinfo.value)
        assert error_msg in expected_error_message
