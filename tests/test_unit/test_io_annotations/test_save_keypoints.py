"""Test saving keypoints annotations to file formats."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from ethology.io.annotations.save_keypoints import (
    _build_sleap_objects,
    _get_image_id_maps,
    _get_keypoint_names,
    _require_sleap_io,
    to_file,
)

# ============================================================================
# Helper Functions for Testing
# ============================================================================


def create_valid_keypoints_dataset(
    n_images: int = 2,
    n_keypoints: int = 2,
    n_instances: int = 1,
    include_confidence: bool = False,
    include_visibility: bool = False,
) -> xr.Dataset:
    """Create a valid keypoints dataset for testing."""
    position_data = np.random.rand(n_images, 2, n_keypoints, n_instances) * 100

    data_vars = {
        "position": (
            ["image_id", "space", "keypoint", "id"],
            position_data,
        ),
    }

    if include_confidence:
        data_vars["confidence"] = (
            ["image_id", "keypoint", "id"],
            np.random.rand(n_images, n_keypoints, n_instances),
        )

    if include_visibility:
        data_vars["visibility"] = (
            ["image_id", "keypoint", "id"],
            np.random.rand(n_images, n_keypoints, n_instances),
        )

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={
            "image_id": np.arange(n_images),
            "space": ["x", "y"],
            "keypoint": [f"kp_{i}" for i in range(n_keypoints)],
            "id": np.arange(n_instances),
        },
    )

    # Add required attributes
    ds.attrs = {
        "annotation_format": "SLEAP",
        "map_keypoint_to_str": {
            i: f"keypoint_{i}" for i in range(n_keypoints)
        },
        "map_image_id_to_filename": {
            i: f"frame_{i}.png" for i in range(n_images)
        },
        "map_image_id_to_video": {
            i: f"video_{i}.mp4" for i in range(n_images)
        },
        "map_image_id_to_frame_idx": {i: i for i in range(n_images)},
    }

    return ds


def test_require_sleap_io_import_success():
    """Test successful import of sleap_io when installed."""
    try:
        sio = _require_sleap_io()
        assert sio is not None
        assert hasattr(sio, "load_file")
    except ModuleNotFoundError:
        pytest.skip("sleap-io not installed")


def test_require_sleap_io_import_missing():
    """Test that ModuleNotFoundError is raised when sleap_io missing."""
    with (
        patch.dict("sys.modules", {"sleap_io": None}),
        pytest.raises(ModuleNotFoundError, match="sleap-io is required"),
    ):
        _require_sleap_io()


def test_get_keypoint_names_from_map():
    """Test extraction of keypoint names from map_keypoint_to_str."""
    ds = create_valid_keypoints_dataset(n_keypoints=3)
    names = _get_keypoint_names(ds)
    assert len(names) == 3
    assert names == ["keypoint_0", "keypoint_1", "keypoint_2"]


def test_get_keypoint_names_from_coordinates():
    """Test extraction of keypoint names from coordinates."""
    ds = create_valid_keypoints_dataset(n_keypoints=2)
    del ds.attrs["map_keypoint_to_str"]

    names = _get_keypoint_names(ds)
    assert len(names) == 2
    assert all(isinstance(name, str) for name in names)


def test_get_keypoint_names_empty_map():
    """Test fallback when map_keypoint_to_str is empty."""
    ds = create_valid_keypoints_dataset(n_keypoints=2)
    ds.attrs["map_keypoint_to_str"] = {}

    names = _get_keypoint_names(ds)
    assert len(names) == 2
    assert all(isinstance(name, str) for name in names)


def test_get_image_id_maps_all_present():
    """Test extraction of all image ID mapping attributes."""
    ds = create_valid_keypoints_dataset()
    maps = _get_image_id_maps(ds)

    assert len(maps) == 3
    assert len(maps[0]) == 2  # filename
    assert len(maps[1]) == 2  # video
    assert len(maps[2]) == 2  # frame_idx


def test_get_image_id_maps_missing_attributes():
    """Test handling of missing mapping attributes."""
    ds = create_valid_keypoints_dataset()
    del ds.attrs["map_image_id_to_video"]

    maps = _get_image_id_maps(ds)
    assert len(maps[1]) == 0  # Video map empty


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
def test_build_sleap_objects_basic_structure(mock_sio):
    """Test basic structure of SLEAP objects built from dataset."""
    ds = create_valid_keypoints_dataset(n_images=1, n_keypoints=2)

    # Mock return values for sleap classes
    mock_module = mock_sio.return_value
    mock_module.Labels.return_value = MagicMock()

    labels = _build_sleap_objects(ds)
    assert labels is not None
    # Ensure Labels constructor was called
    mock_module.Labels.assert_called_once()


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
def test_build_sleap_objects_missing_video_info_error(mock_sio):
    """Test error when video/filename info is missing."""
    ds = create_valid_keypoints_dataset()
    del ds.attrs["map_image_id_to_video"]
    del ds.attrs["map_image_id_to_filename"]

    with pytest.raises(ValueError, match="Missing video or filename"):
        _build_sleap_objects(ds)


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
def test_build_sleap_objects_skips_all_nan_instances(mock_sio):
    """Test that instances with all NaN coordinates are skipped."""
    ds = create_valid_keypoints_dataset(
        n_images=1, n_keypoints=2, n_instances=2
    )
    # Set second instance to all NaN
    ds["position"].values[0, :, :, 1] = np.nan

    _build_sleap_objects(ds)

    # Check that Instance() was instantiated fewer times than total potential
    # instances. We expect 1 instance to be created (the valid one).
    assert mock_sio.return_value.Instance.call_count == 1


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
def test_build_sleap_objects_handles_missing_keypoints(mock_sio):
    """Test handling of missing keypoints (NaN values)."""
    ds = create_valid_keypoints_dataset(n_images=1, n_keypoints=3)
    # Set one keypoint to NaN
    ds["position"].values[0, :, 1, 0] = np.nan

    _build_sleap_objects(ds)

    # Verify Point was called.
    # Total points = 3. One is NaN, so we expect 2 Point creations.
    assert mock_sio.return_value.Point.call_count == 2


def test_to_file_unsupported_format(tmp_path):
    """Test that ValueError is raised for unsupported formats."""
    ds = create_valid_keypoints_dataset()
    output_file = tmp_path / "output.sleap"
    with pytest.raises(ValueError, match="Unsupported format"):
        to_file(ds, output_file, format="INVALID")


def test_to_file_validates_input(tmp_path):
    """Test that to_file validates the input dataset."""
    invalid_ds = xr.Dataset()  # Missing vars
    output_file = tmp_path / "output.sleap"
    # Validator raises ValueError, not TypeError, for missing vars
    with pytest.raises(ValueError):
        to_file(invalid_ds, output_file, format="SLEAP")


@pytest.mark.parametrize(
    "dataset_params",
    [
        {"n_images": 1, "n_keypoints": 2},  # Single Image
        {"n_images": 2, "n_keypoints": 17},  # Many Keypoints
        {"include_confidence": True},  # With Confidence
        {"include_visibility": True},  # With Visibility
    ],
)
@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
@patch("ethology.io.annotations.save_keypoints._build_sleap_objects")
def test_to_file_sleap_variations(
    mock_build, mock_sio, dataset_params, tmp_path
):
    """Test saving to SLEAP format with various dataset configurations."""
    ds = create_valid_keypoints_dataset(**dataset_params)
    output_file = tmp_path / "output.sleap"

    # Mock the internal calls
    mock_build.return_value = MagicMock()
    mock_sio.return_value.save_file = MagicMock()

    result = to_file(ds, output_file, format="SLEAP")

    assert result == output_file
    mock_build.assert_called_once()
    mock_sio.return_value.save_file.assert_called_once()


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
@patch("ethology.io.annotations.save_keypoints._build_sleap_objects")
def test_to_file_output_path_as_string(mock_build, mock_sio, tmp_path):
    """Test that output_filepath can be a string."""
    ds = create_valid_keypoints_dataset()
    output_file = str(tmp_path / "output.sleap")

    # Mock return values so we don't crash on saving
    mock_build.return_value = MagicMock()
    mock_sio.return_value.save_file = MagicMock()

    result = to_file(ds, output_file, format="SLEAP")

    # The code returns input path as-is, so we check equality, not type
    assert result == output_file


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
def test_build_sleap_objects_missing_classes(mock_sio):
    """Test error when sleap-io is missing required classes."""
    ds = create_valid_keypoints_dataset()
    mock_module = mock_sio.return_value
    mock_module.Instance = None

    with pytest.raises(AttributeError, match="sleap-io is missing"):
        _build_sleap_objects(ds)


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
def test_build_sleap_objects_skeleton_fallback(mock_sio):
    """Test Skeleton creation falls back when edges kwarg not supported."""
    ds = create_valid_keypoints_dataset(n_images=1, n_keypoints=1)
    mock_module = mock_sio.return_value

    # First call with edges=[] raises TypeError, second without works
    mock_skeleton = MagicMock()
    mock_module.Skeleton.side_effect = [TypeError, mock_skeleton]
    mock_module.Labels.return_value = MagicMock()

    labels = _build_sleap_objects(ds)
    assert labels is not None
    assert mock_module.Skeleton.call_count == 2


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
def test_build_sleap_objects_video_fallbacks(mock_sio):
    """Test Video construction fallback chain."""
    ds = create_valid_keypoints_dataset(n_images=1, n_keypoints=1)
    mock_module = mock_sio.return_value
    mock_module.Labels.return_value = MagicMock()

    # from_filename raises AttributeError, then filename= raises TypeError
    mock_video = MagicMock()
    mock_module.Video.from_filename.side_effect = AttributeError
    mock_module.Video.side_effect = [TypeError, mock_video]

    _build_sleap_objects(ds)
    assert mock_module.Video.call_count == 2


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
def test_build_sleap_objects_video_cache(mock_sio):
    """Test that videos are reused across frames from the same source."""
    ds = create_valid_keypoints_dataset(n_images=2, n_keypoints=1)
    # Both images map to the same video
    ds.attrs["map_image_id_to_video"] = {0: "shared.mp4", 1: "shared.mp4"}
    mock_module = mock_sio.return_value
    mock_module.Labels.return_value = MagicMock()

    _build_sleap_objects(ds)
    # Video constructor called only once for the shared filename
    mock_module.Video.from_filename.assert_called_once()


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
def test_build_sleap_objects_labeled_frame_fallback(mock_sio):
    """Test LabeledFrame creation falls back to positional args."""
    ds = create_valid_keypoints_dataset(n_images=1, n_keypoints=1)
    mock_module = mock_sio.return_value
    mock_module.Labels.return_value = MagicMock()

    mock_lf = MagicMock()
    mock_module.LabeledFrame.side_effect = [TypeError, mock_lf]

    _build_sleap_objects(ds)
    assert mock_module.LabeledFrame.call_count == 2


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
def test_build_sleap_objects_with_confidence_and_visibility(mock_sio):
    """Test point creation with confidence and visibility data."""
    ds = create_valid_keypoints_dataset(
        n_images=1,
        n_keypoints=2,
        n_instances=1,
        include_confidence=True,
        include_visibility=True,
    )
    mock_module = mock_sio.return_value
    mock_module.Labels.return_value = MagicMock()

    _build_sleap_objects(ds)

    # Points should be created with score and visible kwargs
    point_calls = mock_module.Point.call_args_list
    assert len(point_calls) == 2
    for call in point_calls:
        assert "score" in call.kwargs or "x" in call.kwargs


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
def test_build_sleap_objects_no_point_cls(mock_sio):
    """Test fallback when Point class is not available in sleap-io."""
    ds = create_valid_keypoints_dataset(n_images=1, n_keypoints=1)
    mock_module = mock_sio.return_value
    mock_module.Point = None
    mock_module.Labels.return_value = MagicMock()

    _build_sleap_objects(ds)

    # Instance should be created with list-of-lists points
    inst_call = mock_module.Instance.call_args
    points_arg = inst_call.kwargs.get("points") or inst_call.args[0]
    assert isinstance(points_arg[0], list)


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
def test_build_sleap_objects_instance_fallback(mock_sio):
    """Test Instance creation falls back to positional args."""
    ds = create_valid_keypoints_dataset(n_images=1, n_keypoints=1)
    mock_module = mock_sio.return_value
    mock_module.Labels.return_value = MagicMock()

    mock_inst = MagicMock()
    mock_module.Instance.side_effect = [TypeError, mock_inst]

    _build_sleap_objects(ds)
    assert mock_module.Instance.call_count == 2


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
def test_build_sleap_objects_labels_fallback(mock_sio):
    """Test Labels creation falls back when skeletons kwarg not supported."""
    ds = create_valid_keypoints_dataset(n_images=1, n_keypoints=1)
    mock_module = mock_sio.return_value

    mock_labels = MagicMock()
    mock_module.Labels.side_effect = [TypeError, mock_labels]

    _build_sleap_objects(ds)
    assert mock_module.Labels.call_count == 2


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
@patch("ethology.io.annotations.save_keypoints._build_sleap_objects")
def test_to_file_save_fallback(mock_build, mock_sio, tmp_path):
    """Test save_file falls back to swapped argument order."""
    ds = create_valid_keypoints_dataset()
    output_file = tmp_path / "output.sleap"

    mock_build.return_value = MagicMock()
    mock_module = mock_sio.return_value
    mock_module.save_file.side_effect = [TypeError, None]

    result = to_file(ds, output_file, format="SLEAP")
    assert result == output_file
    assert mock_module.save_file.call_count == 2
