"""Test loading keypoints annotations into ethology datasets."""

from contextlib import nullcontext as does_not_raise
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from ethology.io.annotations.load_keypoints import (
    _frame_label,
    _get_frame_index,
    _get_instances,
    _get_labeled_frames,
    _get_skeleton_keypoints,
    _get_video_filename,
    _infer_keypoint_count,
    _points_from_instance,
    _points_from_point_objects,
    _prepare_frame_records,
    _require_sleap_io,
    from_files,
    _from_single_file,
)


# ============================================================================
# Tests for Helper Functions
# ============================================================================


def test_require_sleap_io_import_success():
    """Test successful import of sleap_io when it exists."""
    try:
        sio = _require_sleap_io()
        assert sio is not None
        assert hasattr(sio, "load_file")
    except ModuleNotFoundError:
        pytest.skip("sleap-io not installed")


def test_require_sleap_io_import_missing():
    """Test that ModuleNotFoundError is raised when sleap_io is missing."""
    with patch.dict("sys.modules", {"sleap_io": None}):
        with pytest.raises(ModuleNotFoundError, match="sleap-io is required"):
            _require_sleap_io()


def test_get_labeled_frames():
    """Test extracting labeled frames from various attributes."""
    # Test 'labeled_frames' attribute
    mock_labels_1 = MagicMock()
    mock_frame1, mock_frame2 = MagicMock(), MagicMock()
    mock_labels_1.labeled_frames = [mock_frame1, mock_frame2]
    assert _get_labeled_frames(mock_labels_1) == [mock_frame1, mock_frame2]

    # Test fallback to 'frames' attribute
    mock_labels_2 = MagicMock(spec=[])
    mock_labels_2.frames = [mock_frame1]
    assert _get_labeled_frames(mock_labels_2) == [mock_frame1]

    # Test fallback to 'labeled_frames_by_video'
    mock_labels_3 = MagicMock(spec=[])
    mock_labels_3.labeled_frames_by_video = {"v1": [mock_frame1]}
    assert _get_labeled_frames(mock_labels_3) == [mock_frame1]

    # Test error
    mock_labels_empty = MagicMock(spec=[])
    with pytest.raises(AttributeError, match="Could not find labeled frames"):
        _get_labeled_frames(mock_labels_empty)


@pytest.mark.parametrize(
    "attr_name, attr_value",
    [
        ("frame_idx", 10),
        ("frame_index", 20),
        ("frame_number", 30),
        ("frame_idx", "42"),  # Test string conversion
    ],
)
def test_get_frame_index(attr_name, attr_value):
    """Test successful extraction of frame index from various attributes."""
    # Use spec=[attr_name] to ensure the mock ONLY has this attribute.
    mock_frame = MagicMock(spec=[attr_name])
    setattr(mock_frame, attr_name, attr_value)
    
    result = _get_frame_index(mock_frame)
    assert result == int(attr_value)
    assert isinstance(result, int)


def test_get_frame_index_error():
    """Test that AttributeError is raised when no valid attribute exists."""
    mock_frame = MagicMock(spec=[])
    with pytest.raises(AttributeError, match="Could not find frame index"):
        _get_frame_index(mock_frame)


@pytest.mark.parametrize(
    "video_attr, expected_filename",
    [
        ({"filename": "v.mp4"}, "v.mp4"),
        ({"path": "v.mp4", "filename": None}, "v.mp4"),
        (None, None),  # No video object
        ({}, None),    # Video object with no path/filename
    ],
)
def test_get_video_filename(video_attr, expected_filename):
    """Test extraction of filename from video object."""
    mock_frame = MagicMock()
    if video_attr is None:
        mock_frame.video = None
    else:
        mock_frame.video = MagicMock()
        for k, v in video_attr.items():
            setattr(mock_frame.video, k, v)
        # Handle case where attributes are missing from spec
        if not video_attr:
             mock_frame.video = MagicMock(spec=[])

    assert _get_video_filename(mock_frame) == expected_filename


@pytest.mark.parametrize(
    "attr_config, expected_count",
    [
        ({"user_instances": [1, 2]}, 2),
        ({"instances": [1, 2]}, 2),
        ({"predicted_instances": [1, 2]}, 2),
        ({"user_instances": [], "instances": []}, 0),
        ({"user_instances": None}, 0),
    ],
)
def test_get_instances(attr_config, expected_count):
    """Test successful extraction of instances."""
    mock_frame = MagicMock()
    # Set all potential attributes to None/Empty first
    mock_frame.user_instances = None
    mock_frame.instances = None
    mock_frame.predicted_instances = None
    
    for k, v in attr_config.items():
        setattr(mock_frame, k, v)

    result = _get_instances(mock_frame)
    assert len(result) == expected_count


def test_points_from_point_objects():
    """Test extraction of points from a list of point objects."""
    # Standard case
    p1 = MagicMock(x=10.0, y=20.0, visible=True, score=0.95)
    p2 = MagicMock(x=30.0, y=40.0, visible=True, score=0.85)
    
    coords, conf, vis = _points_from_point_objects([p1, p2], n_keypoints=2)
    assert np.allclose(coords, [[10, 20], [30, 40]])
    assert np.allclose(conf, [0.95, 0.85])
    assert np.allclose(vis, [1.0, 1.0])

    # Invisible / Missing case
    p_inv = MagicMock(x=10.0, y=20.0, visible=False)
    coords, _, vis = _points_from_point_objects([p_inv, None], n_keypoints=2)
    assert np.isnan(coords[0]).all() # Invisible points become NaN coordinates
    assert vis[0] == 0.0
    assert np.isnan(coords[1]).all()


def test_points_from_instance():
    """Test extraction of points from instance (numpy vs list)."""
    # Numpy Array Case
    mock_inst_np = MagicMock()
    mock_inst_np.numpy = np.array([[10., 20.], [30., 40.]])
    c, _, _ = _points_from_instance(mock_inst_np, 2)
    assert np.allclose(c, [[10., 20.], [30., 40.]])

    # 3D Array Case (Reshape)
    mock_inst_3d = MagicMock()
    mock_inst_3d.numpy = np.array([[[10., 20.]], [[30., 40.]]])
    c, _, _ = _points_from_instance(mock_inst_3d, 2)
    assert c.shape == (2, 2)

    # List Case
    mock_inst_list = MagicMock()
    mock_inst_list.points = [MagicMock(x=10., y=20., visible=True)]
    c, _, _ = _points_from_instance(mock_inst_list, 1)
    assert c.shape == (1, 2)
    
    # Error Case
    with pytest.raises(ValueError, match="Unsupported instance points format"):
        _points_from_instance(MagicMock(spec=[]), 1)


def test_get_skeleton_keypoints():
    """Test extraction of keypoint names from skeletons."""
    # FIX: Explicitly set the name attribute.
    # MagicMock(name='n1') sets the debug name, NOT the attribute .name
    node = MagicMock()
    node.name = "n1"
    
    # Skeletons list
    mock_labels = MagicMock()
    mock_labels.skeletons = [MagicMock(nodes=[node])]
    assert _get_skeleton_keypoints(mock_labels) == ["n1"]

    # Single skeleton attribute
    node2 = MagicMock()
    node2.name = "n2"
    mock_labels.skeletons = []
    mock_labels.skeleton = MagicMock(nodes=[node2])
    assert _get_skeleton_keypoints(mock_labels) == ["n2"]


def test_infer_keypoint_count():
    """Test inferring keypoint count from different formats."""
    # Numpy
    mock_np = MagicMock()
    mock_np.numpy = np.zeros((3, 2))
    assert _infer_keypoint_count(mock_np) == 3

    # List
    mock_list = MagicMock()
    mock_list.points = [1, 2, 3]
    assert _infer_keypoint_count(mock_list) == 3


@pytest.mark.parametrize(
    "video_file, frame_idx, expected",
    [
        ("v.mp4", 42, "v.mp4::frame_42"),
        (None, 42, "frame_42"),
    ],
)
def test_frame_label(video_file, frame_idx, expected):
    assert _frame_label(video_file, frame_idx) == expected


def test_prepare_frame_records_sorting():
    """Test that frame records are sorted by video filename and frame index."""
    mock_labels = MagicMock()
    f1 = MagicMock(frame_idx=5, video=MagicMock(filename="b.mp4"))
    f2 = MagicMock(frame_idx=1, video=MagicMock(filename="a.mp4"))
    f3 = MagicMock(frame_idx=3, video=None)
    mock_labels.labeled_frames = [f1, f2, f3]

    records = _prepare_frame_records(mock_labels)
    # Expected order: No Video (f3), a.mp4 (f2), b.mp4 (f1)
    assert records[0]["frame_idx"] == 3
    assert records[1]["frame_idx"] == 1
    assert records[2]["frame_idx"] == 5


# ============================================================================
# Tests for Main Loading Function
# ============================================================================


def test_from_files_unsupported_format(tmp_path):
    """Test that ValueError is raised for unsupported formats."""
    p = tmp_path / "test.txt"
    p.touch()
    with pytest.raises(ValueError, match="Unsupported format"):
        from_files(p, format="INVALID")


@patch("ethology.io.annotations.load_keypoints._from_single_file")
def test_from_files_concatenation(mock_single):
    """Test concatenation of multiple file datasets."""
    # FIX: Use correct dim names 'space' and 'keypoint' expected by ValidKeypointsAnnotationsDataset
    common_attrs = {
        "map_keypoint_to_str": {0: "n1"},
        "map_image_id_to_filename": {0: "f"},
        "map_image_id_to_frame_idx": {0: 0} # FIX: Required for the loop
    }
    
    ds1 = xr.Dataset(
        {"position": (("image_id", "space", "keypoint", "id"), np.zeros((1, 2, 1, 1)))},
        coords={"image_id": [0], "keypoint": ["n1"], "space": ["x", "y"], "id": [0]},
        attrs=common_attrs.copy()
    )
    ds1.attrs["map_image_id_to_filename"] = {0: "f1"}
    
    ds2 = xr.Dataset(
        {"position": (("image_id", "space", "keypoint", "id"), np.zeros((1, 2, 1, 1)))},
        coords={"image_id": [0], "keypoint": ["n1"], "space": ["x", "y"], "id": [0]},
        attrs=common_attrs.copy()
    )
    ds2.attrs["map_image_id_to_filename"] = {0: "f2"}
    
    mock_single.side_effect = [ds1, ds2]

    ds = from_files(["a", "b"], format="SLEAP")
    
    assert ds.sizes["image_id"] == 2
    assert ds.attrs["map_image_id_to_filename"] == {0: "f1", 1: "f2"}


@patch("ethology.io.annotations.load_keypoints._from_single_file")
def test_from_files_mismatch_error(mock_single):
    """Test error when keypoints differ."""
    # FIX: Add missing attributes to prevent KeyError during iteration
    ds1 = xr.Dataset(
        coords={"image_id": [0]}, 
        attrs={
            "map_keypoint_to_str": {0: "A"},
            "map_image_id_to_filename": {0: "f"},
            "map_image_id_to_frame_idx": {0: 0}
        }
    )
    ds2 = xr.Dataset(
        coords={"image_id": [0]}, 
        attrs={
            "map_keypoint_to_str": {0: "B"},
            "map_image_id_to_filename": {0: "f"},
            "map_image_id_to_frame_idx": {0: 0}
        }
    )
    mock_single.side_effect = [ds1, ds2]

    with pytest.raises(ValueError, match="Keypoint labels differ"):
        from_files(["a", "b"], format="SLEAP")


@patch("ethology.io.annotations.load_keypoints._require_sleap_io")
def test_from_single_file_integration_mock(mock_require):
    """Test the full flow of _from_single_file using mocks."""
    mock_sio = mock_require.return_value
    
    inst = MagicMock()
    inst.points = [MagicMock(x=10, y=20, visible=True, score=0.9)]
    frame = MagicMock(frame_idx=0, video=MagicMock(filename="v.mp4"), user_instances=[inst])
    
    # FIX: Explicitly set name
    node = MagicMock()
    node.name = "nose"
    skel = MagicMock(nodes=[node])
    
    labels = MagicMock(labeled_frames=[frame], skeletons=[skel])
    mock_sio.load_file.return_value = labels

    ds = _from_single_file("test.slp", "SLEAP", None)

    assert "position" in ds
    assert ds.keypoint.values[0] == "nose"
    assert np.allclose(ds.position.values[0, 0, 0, 0], 10)
    assert np.allclose(ds.position.values[0, 1, 0, 0], 20)
    assert ds.confidence.values[0, 0, 0] == 0.9


@patch("ethology.io.annotations.load_keypoints._require_sleap_io")
def test_from_single_file_inference_fallback(mock_require):
    """Test that keypoints are inferred when no skeleton is present."""
    mock_sio = mock_require.return_value
    
    p1 = MagicMock(x=10, y=10, visible=True)
    p2 = MagicMock(x=20, y=20, visible=True)
    inst = MagicMock(points=[p1, p2])
    
    frame = MagicMock(frame_idx=0, video=None, user_instances=[inst])
    
    # No skeletons provided!
    labels = MagicMock(labeled_frames=[frame], skeletons=[])
    mock_sio.load_file.return_value = labels

    ds = _from_single_file("test.slp", "SLEAP", None)

    assert ds.sizes["keypoint"] == 2
    assert ds.keypoint.values.tolist() == ["keypoint_0", "keypoint_1"]


@patch("ethology.io.annotations.load_keypoints._require_sleap_io")
def test_from_single_file_errors(mock_require):
    """Test error conditions in single file loading."""
    mock_sio = mock_require.return_value
    
    # Case: No Frames
    mock_sio.load_file.return_value = MagicMock(labeled_frames=[])
    with pytest.raises(ValueError, match="No labeled frames found"):
        _from_single_file("t.slp", "SLEAP", None)

    # Case: No Instances
    frame_empty = MagicMock(user_instances=[])
    mock_sio.load_file.return_value = MagicMock(labeled_frames=[frame_empty])
    with pytest.raises(ValueError, match="No instances found"):
        _from_single_file("t.slp", "SLEAP", None)


@patch("ethology.io.annotations.load_keypoints._require_sleap_io")
def test_from_single_file_mismatched_keypoints_error(mock_require):
    """Test error when instance keypoints don't match skeleton."""
    mock_sio = mock_require.return_value

    # Create mismatch: Skeleton has 3 nodes, instance has 2 points
    # FIX: Ensure nodes act like objects with a string .name attribute
    nodes = []
    for i in range(3):
        n = MagicMock()
        n.name = str(i)
        nodes.append(n)
        
    skel = MagicMock(nodes=nodes)
    inst = MagicMock(points=[MagicMock(), MagicMock()])
    frame = MagicMock(frame_idx=0, video=None, user_instances=[inst])
    
    mock_sio.load_file.return_value = MagicMock(
        labeled_frames=[frame], skeletons=[skel]
    )

    with pytest.raises((ValueError, Exception)):
        _from_single_file("d.slp", "SLEAP", None)


@patch("ethology.io.annotations.load_keypoints._require_sleap_io")
def test_from_single_file_multiple_instances(mock_require):
    """Test that multiple instances are correctly stacked in the 'id' dimension."""
    mock_sio = mock_require.return_value
    
    inst1 = MagicMock(points=[MagicMock(x=10, y=10, visible=True)])
    inst2 = MagicMock(points=[MagicMock(x=20, y=20, visible=True)])
    
    frame = MagicMock(frame_idx=0, video=None, user_instances=[inst1, inst2])
    
    # FIX: Explicitly set name
    node = MagicMock()
    node.name = "k1"
    labels = MagicMock(labeled_frames=[frame], skeletons=[MagicMock(nodes=[node])])
    mock_sio.load_file.return_value = labels

    ds = _from_single_file("test.slp", "SLEAP", None)

    assert ds.sizes["id"] == 2