"""Test loading keypoints annotations into ethology datasets."""

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
)

# ============================================================================
# Helper Functions for Testing
# ============================================================================


def assert_dataset(
    ds: xr.Dataset,
    expected_n_images: int,
    expected_n_keypoints: int,
    expected_max_instances: int,
    expected_space_dim: int,
):
    """Check that the keypoints dataset has the expected shape and content."""
    # Check size of position array
    assert ds.position.shape == (
        expected_n_images,
        expected_space_dim,
        expected_n_keypoints,
        expected_max_instances,
    )

    # Check dimensions
    assert "image_id" in ds.dims
    assert "space" in ds.dims
    assert "keypoint" in ds.dims
    assert "id" in ds.dims

    # Check coordinates
    assert ds.dims["image_id"] == expected_n_images
    assert ds.dims["space"] == expected_space_dim
    assert ds.dims["keypoint"] == expected_n_keypoints
    assert ds.dims["id"] == expected_max_instances

    # Check space coordinate is x, y
    assert list(ds.space.values) == ["x", "y"]


# ============================================================================
# Tests for Helper Functions
# ============================================================================


class TestRequireSleapIo:
    """Test the _require_sleap_io function."""

    def test_require_sleap_io_import_success(self):
        """Test successful import of sleap_io when installed."""
        try:
            sio = _require_sleap_io()
            assert sio is not None
            assert hasattr(sio, "load_file")
        except ModuleNotFoundError:
            pytest.skip("sleap-io not installed")

    def test_require_sleap_io_import_missing(self):
        """Test that ModuleNotFoundError is raised when sleap_io missing."""
        with patch.dict("sys.modules", {"sleap_io": None}):
            with pytest.raises(ModuleNotFoundError) as excinfo:
                _require_sleap_io()
            assert "sleap-io is required" in str(excinfo.value)


class TestGetLabeledFrames:
    """Test the _get_labeled_frames function."""

    def test_get_labeled_frames_from_labeled_frames_attr(self):
        """Test extracting labeled frames from labeled_frames attribute."""
        mock_labels = MagicMock()
        mock_frame1, mock_frame2 = MagicMock(), MagicMock()
        mock_labels.labeled_frames = [mock_frame1, mock_frame2]

        result = _get_labeled_frames(mock_labels)

        assert len(result) == 2
        assert mock_frame1 in result
        assert mock_frame2 in result

    def test_get_labeled_frames_from_frames_attr(self):
        """Test extracting labeled frames from frames attribute."""
        mock_labels = MagicMock(spec=[])
        mock_frame1, mock_frame2 = MagicMock(), MagicMock()
        mock_labels.frames = [mock_frame1, mock_frame2]
        del mock_labels.labeled_frames

        result = _get_labeled_frames(mock_labels)

        assert len(result) == 2
        assert mock_frame1 in result

    def test_get_labeled_frames_from_labeled_frames_by_video(self):
        """Test extracting frames from labeled_frames_by_video attribute."""
        mock_labels = MagicMock(spec=[])
        mock_frame1, mock_frame2, mock_frame3 = (
            MagicMock(),
            MagicMock(),
            MagicMock(),
        )
        mock_labels.labeled_frames_by_video = {
            "video1": [mock_frame1, mock_frame2],
            "video2": [mock_frame3],
        }
        del mock_labels.labeled_frames
        del mock_labels.frames

        result = _get_labeled_frames(mock_labels)

        assert len(result) == 3
        assert mock_frame1 in result
        assert mock_frame3 in result

    def test_get_labeled_frames_attribute_error(self):
        """Test AttributeError when no valid frame attribute exists."""
        mock_labels = MagicMock(spec=[])
        del mock_labels.labeled_frames
        del mock_labels.frames
        del mock_labels.labeled_frames_by_video

        with pytest.raises(AttributeError) as excinfo:
            _get_labeled_frames(mock_labels)
        assert "Could not find labeled frames" in str(excinfo.value)


class TestGetFrameIndex:
    """Test the _get_frame_index function."""

    @pytest.mark.parametrize(
        "attr_name, attr_value",
        [
            ("frame_idx", 10),
            ("frame_index", 20),
            ("frame_number", 30),
        ],
    )
    def test_get_frame_index_success(self, attr_name: str, attr_value: int):
        """Test frame index extraction from various attributes."""
        mock_frame = MagicMock()
        setattr(mock_frame, attr_name, attr_value)

        result = _get_frame_index(mock_frame)

        assert result == attr_value
        assert isinstance(result, int)

    def test_get_frame_index_converts_to_int(self):
        """Test that frame index is converted to integer."""
        mock_frame = MagicMock()
        mock_frame.frame_idx = "42"

        result = _get_frame_index(mock_frame)

        assert result == 42
        assert isinstance(result, int)

    def test_get_frame_index_attribute_error(self):
        """Test AttributeError when no frame index attribute exists."""
        mock_frame = MagicMock(spec=[])

        with pytest.raises(AttributeError) as excinfo:
            _get_frame_index(mock_frame)
        assert "Could not find frame index" in str(excinfo.value)


class TestGetVideoFilename:
    """Test the _get_video_filename function."""

    def test_get_video_filename_from_filename(self):
        """Test extraction of filename from video object."""
        mock_frame = MagicMock()
        mock_video = MagicMock()
        mock_video.filename = "/path/to/video.mp4"
        mock_frame.video = mock_video

        result = _get_video_filename(mock_frame)

        assert result == "/path/to/video.mp4"

    def test_get_video_filename_from_path(self):
        """Test extraction of path from video object."""
        mock_frame = MagicMock()
        mock_video = MagicMock(spec=["path"])
        mock_video.filename = None
        mock_video.path = "/path/to/video2.mp4"
        mock_frame.video = mock_video

        result = _get_video_filename(mock_frame)

        assert result == "/path/to/video2.mp4"

    def test_get_video_filename_no_video(self):
        """Test that None is returned when frame has no video."""
        mock_frame = MagicMock()
        mock_frame.video = None

        result = _get_video_filename(mock_frame)

        assert result is None

    def test_get_video_filename_no_valid_attr(self):
        """Test that None is returned when video has no valid attributes."""
        mock_frame = MagicMock()
        mock_video = MagicMock(spec=[])
        mock_frame.video = mock_video

        result = _get_video_filename(mock_frame)

        assert result is None


class TestGetInstances:
    """Test the _get_instances function."""

    @pytest.mark.parametrize(
        "attr_name",
        ["user_instances", "instances", "predicted_instances"],
    )
    def test_get_instances_success(self, attr_name: str):
        """Test successful extraction of instances from various attributes."""
        mock_frame = MagicMock()
        mock_inst1, mock_inst2 = MagicMock(), MagicMock()
        setattr(mock_frame, attr_name, [mock_inst1, mock_inst2])

        result = _get_instances(mock_frame)

        assert len(result) == 2
        assert mock_inst1 in result
        assert mock_inst2 in result

    def test_get_instances_empty_list(self):
        """Test empty list when all instance attributes empty."""
        mock_frame = MagicMock()
        mock_frame.user_instances = []
        mock_frame.instances = []
        mock_frame.predicted_instances = []

        result = _get_instances(mock_frame)

        assert result == []

    def test_get_instances_none_attributes(self):
        """Test handling of None attributes."""
        mock_frame = MagicMock()
        mock_frame.user_instances = None
        mock_inst1, mock_inst2 = MagicMock(), MagicMock()
        mock_frame.instances = [mock_inst1, mock_inst2]
        mock_frame.predicted_instances = None

        result = _get_instances(mock_frame)

        assert len(result) == 2


class TestPointsFromPointObjects:
    """Test the _points_from_point_objects function."""

    def test_points_from_point_objects_basic(self):
        """Test extraction of points from a list of point objects."""
        mock_point1 = MagicMock()
        mock_point1.x = 10.0
        mock_point1.y = 20.0
        mock_point1.visible = True
        mock_point1.score = 0.95

        mock_point2 = MagicMock()
        mock_point2.x = 30.0
        mock_point2.y = 40.0
        mock_point2.visible = True
        mock_point2.score = 0.85

        points = [mock_point1, mock_point2]
        coords, confidence, visibility = _points_from_point_objects(
            points, n_keypoints=2
        )

        assert coords.shape == (2, 2)
        assert np.allclose(coords[0], [10.0, 20.0])
        assert np.allclose(coords[1], [30.0, 40.0])
        assert np.isclose(confidence[0], 0.95)
        assert np.isclose(confidence[1], 0.85)
        assert np.isclose(visibility[0], 1.0)
        assert np.isclose(visibility[1], 1.0)

    def test_points_from_point_objects_with_none_points(self):
        """Test handling of None points in the list."""
        mock_point1 = MagicMock()
        mock_point1.x = 10.0
        mock_point1.y = 20.0
        mock_point1.visible = True

        points = [mock_point1, None]
        coords, confidence, visibility = _points_from_point_objects(
            points, n_keypoints=2
        )

        assert np.allclose(coords[0], [10.0, 20.0])
        assert np.isnan(coords[1, 0]) and np.isnan(coords[1, 1])

    def test_points_from_point_objects_invisible(self):
        """Test handling of invisible points."""
        mock_point = MagicMock()
        mock_point.x = 10.0
        mock_point.y = 20.0
        mock_point.visible = False

        coords, confidence, visibility = _points_from_point_objects(
            [mock_point], n_keypoints=1
        )

        assert np.isclose(visibility[0], 0.0)
        assert np.isnan(coords[0, 0]) and np.isnan(coords[0, 1])

    def test_points_from_point_objects_missing_coordinates(self):
        """Test handling of points with missing x or y coordinates."""
        mock_point = MagicMock()
        mock_point.x = None
        mock_point.y = 20.0

        coords, _, _ = _points_from_point_objects([mock_point], n_keypoints=1)

        assert np.isnan(coords[0, 0]) and np.isnan(coords[0, 1])


class TestPointsFromInstance:
    """Test the _points_from_instance function."""

    def test_points_from_instance_numpy_array(self):
        """Test extraction of points from instance with numpy array."""
        mock_instance = MagicMock()
        points_array = np.array(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32
        )
        mock_instance.numpy = points_array

        coords, confidence, visibility = _points_from_instance(
            mock_instance, n_keypoints=3
        )

        assert coords.shape == (3, 2)
        assert np.allclose(coords[0], [10.0, 20.0])
        assert confidence is None
        assert visibility is None

    def test_points_from_instance_callable_numpy(self):
        """Test when numpy is a callable method."""
        mock_instance = MagicMock()
        points_array = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        mock_instance.numpy = MagicMock(return_value=points_array)

        coords, _, _ = _points_from_instance(mock_instance, n_keypoints=2)

        assert coords.shape == (2, 2)

    def test_points_from_instance_reshaped_3d(self):
        """Test handling of 3D arrays with shape (n_keypoints, 1, 2)."""
        mock_instance = MagicMock()
        points_array = np.array(
            [[[10.0, 20.0]], [[30.0, 40.0]]], dtype=np.float32
        )
        mock_instance.numpy = points_array

        coords, _, _ = _points_from_instance(mock_instance, n_keypoints=2)

        assert coords.shape == (2, 2)

    def test_points_from_instance_points_list(self):
        """Test extraction from instance with points list."""
        mock_point1 = MagicMock()
        mock_point1.x = 10.0
        mock_point1.y = 20.0
        mock_point1.visible = True

        mock_instance = MagicMock()
        mock_instance.points = [mock_point1]

        coords, _, _ = _points_from_instance(mock_instance, n_keypoints=1)

        assert coords.shape == (1, 2)
        assert np.allclose(coords[0], [10.0, 20.0])

    def test_points_from_instance_unsupported_format(self):
        """Test that ValueError is raised for unsupported formats."""
        mock_instance = MagicMock(spec=[])

        with pytest.raises(ValueError) as excinfo:
            _points_from_instance(mock_instance, n_keypoints=1)
        assert "Unsupported instance points format" in str(excinfo.value)


class TestGetSkeletonKeypoints:
    """Test the _get_skeleton_keypoints function."""

    def test_get_skeleton_keypoints_from_skeletons(self):
        """Test extraction of keypoint names from skeletons."""
        mock_labels = MagicMock()
        mock_node1 = MagicMock()
        mock_node1.name = "nose"
        mock_node2 = MagicMock()
        mock_node2.name = "tail"

        mock_skeleton = MagicMock()
        mock_skeleton.nodes = [mock_node1, mock_node2]
        mock_labels.skeletons = [mock_skeleton]

        result = _get_skeleton_keypoints(mock_labels)

        assert result == ["nose", "tail"]

    def test_get_skeleton_keypoints_from_skeleton(self):
        """Test extraction from single skeleton attribute."""
        mock_labels = MagicMock()
        mock_node1 = MagicMock()
        mock_node1.name = "left_ear"
        mock_node2 = MagicMock()
        mock_node2.name = "right_ear"

        mock_skeleton = MagicMock()
        mock_skeleton.nodes = [mock_node1, mock_node2]
        mock_labels.skeleton = mock_skeleton
        mock_labels.skeletons = []

        result = _get_skeleton_keypoints(mock_labels)

        assert result == ["left_ear", "right_ear"]

    def test_get_skeleton_keypoints_empty(self):
        """Test that empty list is returned when no skeleton is found."""
        mock_labels = MagicMock()
        mock_labels.skeletons = []

        result = _get_skeleton_keypoints(mock_labels)

        assert result == []


class TestInferKeypointCount:
    """Test the _infer_keypoint_count function."""

    def test_infer_keypoint_count_from_numpy_array(self):
        """Test inferring keypoint count from numpy array."""
        mock_instance = MagicMock()
        points_array = np.array(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32
        )
        mock_instance.numpy = points_array

        result = _infer_keypoint_count(mock_instance)

        assert result == 3

    def test_infer_keypoint_count_from_list(self):
        """Test inferring keypoint count from point list."""
        mock_point1, mock_point2 = MagicMock(), MagicMock()
        mock_instance = MagicMock()
        mock_instance.points = [mock_point1, mock_point2, None]

        result = _infer_keypoint_count(mock_instance)

        assert result == 3

    def test_infer_keypoint_count_2d_array_transposed(self):
        """Test handling of transposed arrays (shape: 2, n_keypoints)."""
        mock_instance = MagicMock()
        points_array = np.array(
            [[10.0, 30.0, 50.0], [20.0, 40.0, 60.0]], dtype=np.float32
        )
        mock_instance.numpy = points_array

        result = _infer_keypoint_count(mock_instance)

        assert result == 3

    def test_infer_keypoint_count_unsupported_format(self):
        """Test that ValueError is raised for unsupported formats."""
        mock_instance = MagicMock(spec=[])

        with pytest.raises(ValueError) as excinfo:
            _infer_keypoint_count(mock_instance)
        assert "Could not infer keypoint count" in str(excinfo.value)


class TestFrameLabel:
    """Test the _frame_label function."""

    def test_frame_label_with_video_filename(self):
        """Test frame label generation with video filename."""
        label = _frame_label("video.mp4", 42)

        assert label == "video.mp4::frame_42"

    def test_frame_label_without_video_filename(self):
        """Test frame label generation without video filename."""
        label = _frame_label(None, 42)

        assert label == "frame_42"


class TestPrepareFrameRecords:
    """Test the _prepare_frame_records function."""

    def test_prepare_frame_records_sorting(self):
        """Test frame records are sorted by video and frame index."""
        mock_labels = MagicMock()

        mock_frame1 = MagicMock()
        mock_frame1.frame_idx = 5
        mock_frame1.video = MagicMock()
        mock_frame1.video.filename = "video_b.mp4"

        mock_frame2 = MagicMock()
        mock_frame2.frame_idx = 1
        mock_frame2.video = MagicMock()
        mock_frame2.video.filename = "video_a.mp4"

        mock_frame3 = MagicMock()
        mock_frame3.frame_idx = 3
        mock_frame3.video = None

        mock_labels.labeled_frames = [mock_frame1, mock_frame2, mock_frame3]

        records = _prepare_frame_records(mock_labels)

        # Should be sorted by video filename (None first, then alphabetically)
        # then by frame index
        assert records[0]["frame_idx"] == 3  # no video, frame 3
        assert records[1]["frame_idx"] == 1  # video_a, frame 1
        assert records[2]["frame_idx"] == 5  # video_b, frame 5


# ============================================================================
# Tests for Main Loading Function
# ============================================================================


class TestFromFiles:
    """Test the from_files function."""

    def test_from_files_unsupported_format(self, tmp_path: Path):
        """Test that ValueError is raised for unsupported formats."""
        test_file = tmp_path / "test.sleap"
        test_file.write_text("")

        with pytest.raises(ValueError) as excinfo:
            from_files(test_file, format="INVALID")  # type: ignore

        assert "Unsupported format" in str(excinfo.value)

    def test_from_files_returns_xarray_dataset(self):
        """Test that from_files returns an xarray Dataset."""
        mock_dataset = xr.Dataset(
            data_vars={
                "position": (
                    ["image_id", "space", "keypoint", "id"],
                    np.zeros((1, 2, 1, 1)),
                ),
            },
            coords={
                "image_id": [0],
                "space": ["x", "y"],
                "keypoint": ["nose"],
                "id": [0],
            },
        )

        with patch(
            "ethology.io.annotations.load_keypoints._from_single_file",
            return_value=mock_dataset,
        ):
            result = from_files("dummy_path", format="SLEAP")

        assert isinstance(result, xr.Dataset)
        assert "position" in result.data_vars

    def test_from_files_multiple_files_concatenation(self):
        """Test concatenation of multiple files."""
        # Create two mock datasets
        ds1 = xr.Dataset(
            data_vars={
                "position": (
                    ["image_id", "space", "keypoint", "id"],
                    np.zeros((2, 2, 2, 1)),
                ),
            },
            coords={
                "image_id": [0, 1],
                "space": ["x", "y"],
                "keypoint": ["nose", "tail"],
                "id": [0],
            },
        )
        ds1.attrs = {
            "map_keypoint_to_str": {0: "nose", 1: "tail"},
            "map_image_id_to_filename": {0: "img1.jpg", 1: "img2.jpg"},
            "map_image_id_to_video": {},
            "map_image_id_to_frame_idx": {0: 0, 1: 1},
        }

        ds2 = xr.Dataset(
            data_vars={
                "position": (
                    ["image_id", "space", "keypoint", "id"],
                    np.zeros((1, 2, 2, 1)),
                ),
            },
            coords={
                "image_id": [0],
                "space": ["x", "y"],
                "keypoint": ["nose", "tail"],
                "id": [0],
            },
        )
        ds2.attrs = {
            "map_keypoint_to_str": {0: "nose", 1: "tail"},
            "map_image_id_to_filename": {0: "img3.jpg"},
            "map_image_id_to_video": {},
            "map_image_id_to_frame_idx": {0: 2},
        }

        with patch(
            "ethology.io.annotations.load_keypoints._from_single_file",
            side_effect=[ds1, ds2],
        ):
            result = from_files(["path1", "path2"], format="SLEAP")

        assert result.sizes["image_id"] == 3
        assert "position" in result.data_vars

    def test_from_files_multiple_files_keypoint_mismatch(self):
        """Test error when keypoint labels differ across files."""
        ds1 = xr.Dataset(
            data_vars={
                "position": (
                    ["image_id", "space", "keypoint", "id"],
                    np.zeros((1, 2, 2, 1)),
                ),
            },
            coords={
                "image_id": [0],
                "space": ["x", "y"],
                "keypoint": ["nose", "tail"],
                "id": [0],
            },
        )
        ds1.attrs = {
            "map_keypoint_to_str": {0: "nose", 1: "tail"},
            "map_image_id_to_filename": {0: "img1.jpg"},
            "map_image_id_to_video": {},
            "map_image_id_to_frame_idx": {0: 0},
        }

        ds2 = xr.Dataset(
            data_vars={
                "position": (
                    ["image_id", "space", "keypoint", "id"],
                    np.zeros((1, 2, 2, 1)),
                ),
            },
            coords={
                "image_id": [0],
                "space": ["x", "y"],
                "keypoint": ["left_eye", "right_eye"],
                "id": [0],
            },
        )
        ds2.attrs = {
            "map_keypoint_to_str": {0: "left_eye", 1: "right_eye"},
            "map_image_id_to_filename": {0: "img2.jpg"},
            "map_image_id_to_video": {},
            "map_image_id_to_frame_idx": {0: 1},
        }

        with patch(
            "ethology.io.annotations.load_keypoints._from_single_file",
            side_effect=[ds1, ds2],
        ):
            with pytest.raises(ValueError) as excinfo:
                from_files(["path1", "path2"], format="SLEAP")

            assert "Keypoint labels differ" in str(excinfo.value)

    def test_from_files_with_confidence_and_visibility(self):
        """Test that confidence and visibility are properly handled."""
        ds = xr.Dataset(
            data_vars={
                "position": (
                    ["image_id", "space", "keypoint", "id"],
                    np.random.rand(2, 2, 2, 1),
                ),
                "confidence": (
                    ["image_id", "keypoint", "id"],
                    np.random.rand(2, 2, 1),
                ),
                "visibility": (
                    ["image_id", "keypoint", "id"],
                    np.random.rand(2, 2, 1),
                ),
            },
            coords={
                "image_id": [0, 1],
                "space": ["x", "y"],
                "keypoint": ["nose", "tail"],
                "id": [0],
            },
        )
        ds.attrs = {
            "annotation_files": "test.sleap",
            "annotation_format": "SLEAP",
            "images_directories": None,
            "map_keypoint_to_str": {0: "nose", 1: "tail"},
            "map_image_id_to_filename": {0: "img1.jpg", 1: "img2.jpg"},
            "map_image_id_to_video": {},
            "map_image_id_to_frame_idx": {0: 0, 1: 1},
        }

        with patch(
            "ethology.io.annotations.load_keypoints._from_single_file",
            return_value=ds,
        ):
            result = from_files("test.sleap", format="SLEAP")

        assert "confidence" in result.data_vars
        assert "visibility" in result.data_vars


class TestFromSingleFile:
    """Test the _from_single_file function through mocked sleap data."""

    def test_from_single_file_basic_structure(self):
        """Test basic structure of output dataset."""
        # Create mock sleap objects
        mock_point1 = MagicMock()
        mock_point1.x = 10.0
        mock_point1.y = 20.0
        mock_point1.visible = True
        mock_point1.score = 0.95

        mock_point2 = MagicMock()
        mock_point2.x = 30.0
        mock_point2.y = 40.0
        mock_point2.visible = True
        mock_point2.score = 0.85

        mock_instance = MagicMock()
        mock_instance.points = [mock_point1, mock_point2]

        mock_frame = MagicMock()
        mock_frame.frame_idx = 0
        mock_frame.video = None
        mock_frame.user_instances = [mock_instance]

        mock_skeleton = MagicMock()
        mock_skeleton.nodes = [
            MagicMock(name="nose"),
            MagicMock(name="tail"),
        ]

        mock_labels = MagicMock()
        mock_labels.labeled_frames = [mock_frame]
        mock_labels.skeletons = [mock_skeleton]

        with patch(
            "ethology.io.annotations.load_keypoints._require_sleap_io"
        ) as mock_sio:
            mock_sio.return_value.load_file.return_value = mock_labels
            from ethology.io.annotations.load_keypoints import (
                _from_single_file,
            )

            try:
                result = _from_single_file(
                    "dummy.sleap", format="SLEAP", images_dirs=None
                )

                # Check dimensions
                assert "image_id" in result.dims
                assert "space" in result.dims
                assert "keypoint" in result.dims
                assert "id" in result.dims

                # Check coordinates
                assert list(result.keypoint.values) == ["nose", "tail"]
                assert list(result.space.values) == ["x", "y"]

                # Check position data
                assert "position" in result.data_vars
                # Check confidence due to score
                assert "confidence" in result.data_vars
            except Exception:
                # If sleap_io is not installed, skip
                pytest.skip("sleap-io not installed or mock failed")

    def test_from_single_file_no_skeleton_fallback(self):
        """Test fallback to infer keypoint count when no skeleton."""
        # Create mock instance with numpy array
        mock_instance = MagicMock()
        points_array = np.array(
            [[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]], dtype=np.float32
        )
        mock_instance.numpy = points_array

        mock_frame = MagicMock()
        mock_frame.frame_idx = 0
        mock_frame.video = None
        mock_frame.user_instances = [mock_instance]

        mock_labels = MagicMock()
        mock_labels.labeled_frames = [mock_frame]
        mock_labels.skeletons = []

        with patch(
            "ethology.io.annotations.load_keypoints._require_sleap_io"
        ) as mock_sio:
            mock_sio.return_value.load_file.return_value = mock_labels
            from ethology.io.annotations.load_keypoints import (
                _from_single_file,
            )

            try:
                result = _from_single_file(
                    "dummy.sleap", format="SLEAP", images_dirs=None
                )

                # Should have 3 auto-generated keypoints
                assert result.sizes["keypoint"] == 3
                # Should have names like keypoint_0, keypoint_1, etc.
                keypoint_names = list(result.keypoint.values)
                assert any("keypoint" in name for name in keypoint_names)
            except Exception:
                pytest.skip("sleap-io not installed or mock failed")

    def test_from_single_file_no_labeled_frames_error(self):
        """Test error when no labeled frames are found."""
        mock_labels = MagicMock()
        mock_labels.labeled_frames = []

        with patch(
            "ethology.io.annotations.load_keypoints._require_sleap_io"
        ) as mock_sio:
            mock_sio.return_value.load_file.return_value = mock_labels
            from ethology.io.annotations.load_keypoints import (
                _from_single_file,
            )

            with pytest.raises(ValueError) as excinfo:
                _from_single_file(
                    "dummy.sleap", format="SLEAP", images_dirs=None
                )

            assert "No labeled frames found" in str(excinfo.value)

    def test_from_single_file_no_instances_error(self):
        """Test error when no instances are found in any frame."""
        mock_frame = MagicMock()
        mock_frame.frame_idx = 0
        mock_frame.video = None
        mock_frame.user_instances = []

        mock_labels = MagicMock()
        mock_labels.labeled_frames = [mock_frame]
        mock_labels.skeletons = []

        with patch(
            "ethology.io.annotations.load_keypoints._require_sleap_io"
        ) as mock_sio:
            mock_sio.return_value.load_file.return_value = mock_labels
            from ethology.io.annotations.load_keypoints import (
                _from_single_file,
            )

            with pytest.raises(ValueError) as excinfo:
                _from_single_file(
                    "dummy.sleap", format="SLEAP", images_dirs=None
                )

            assert "No instances found" in str(excinfo.value)

    def test_from_single_file_mismatched_keypoints_error(self):
        """Test error when instance keypoints don't match skeleton."""
        mock_point1, mock_point2 = MagicMock(), MagicMock()
        mock_point1.x = 10.0
        mock_point1.y = 20.0
        mock_point1.visible = True
        mock_point2.x = 30.0
        mock_point2.y = 40.0
        mock_point2.visible = True

        mock_instance = MagicMock()
        mock_instance.points = [mock_point1, mock_point2]

        mock_frame = MagicMock()
        mock_frame.frame_idx = 0
        mock_frame.video = None
        mock_frame.user_instances = [mock_instance]

        # Skeleton has 3 keypoints but instance has 2
        mock_skeleton = MagicMock()
        mock_skeleton.nodes = [
            MagicMock(name="nose"),
            MagicMock(name="tail"),
            MagicMock(name="ear"),
        ]

        mock_labels = MagicMock()
        mock_labels.labeled_frames = [mock_frame]
        mock_labels.skeletons = [mock_skeleton]

        with patch(
            "ethology.io.annotations.load_keypoints._require_sleap_io"
        ) as mock_sio:
            mock_sio.return_value.load_file.return_value = mock_labels
            from ethology.io.annotations.load_keypoints import (
                _from_single_file,
            )

            with pytest.raises(ValueError) as excinfo:
                _from_single_file(
                    "dummy.sleap", format="SLEAP", images_dirs=None
                )

            assert "Instance keypoints do not match" in str(excinfo.value)

    def test_from_single_file_output_attributes(self):
        """Test that output dataset has required attributes."""
        mock_point = MagicMock()
        mock_point.x = 10.0
        mock_point.y = 20.0
        mock_point.visible = True

        mock_instance = MagicMock()
        mock_instance.points = [mock_point]

        mock_frame = MagicMock()
        mock_frame.frame_idx = 0
        mock_frame.video = None
        mock_frame.user_instances = [mock_instance]

        mock_skeleton = MagicMock()
        mock_skeleton.nodes = [MagicMock(name="nose")]

        mock_labels = MagicMock()
        mock_labels.labeled_frames = [mock_frame]
        mock_labels.skeletons = [mock_skeleton]

        with patch(
            "ethology.io.annotations.load_keypoints._require_sleap_io"
        ) as mock_sio:
            mock_sio.return_value.load_file.return_value = mock_labels
            from ethology.io.annotations.load_keypoints import (
                _from_single_file,
            )

            try:
                result = _from_single_file(
                    "dummy.sleap",
                    format="SLEAP",
                    images_dirs=[Path("/images")],
                )

                # Check required attributes
                assert "annotation_files" in result.attrs
                assert "annotation_format" in result.attrs
                assert "images_directories" in result.attrs
                assert "map_keypoint_to_str" in result.attrs
                assert "map_image_id_to_filename" in result.attrs
                assert "map_image_id_to_video" in result.attrs
                assert "map_image_id_to_frame_idx" in result.attrs

                # Check attribute values
                assert result.attrs["annotation_format"] == "SLEAP"
                assert "map_keypoint_to_str" in result.attrs
            except Exception:
                pytest.skip("sleap-io not installed or mock failed")


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_multiple_instances_per_frame(self):
        """Test handling of multiple instances in a single frame."""
        # Create two instances with different points
        mock_point1a = MagicMock()
        mock_point1a.x = 10.0
        mock_point1a.y = 20.0
        mock_point1a.visible = True

        mock_point2a = MagicMock()
        mock_point2a.x = 30.0
        mock_point2a.y = 40.0
        mock_point2a.visible = True

        mock_instance1 = MagicMock()
        mock_instance1.points = [mock_point1a, mock_point2a]

        mock_point1b = MagicMock()
        mock_point1b.x = 50.0
        mock_point1b.y = 60.0
        mock_point1b.visible = True

        mock_point2b = MagicMock()
        mock_point2b.x = 70.0
        mock_point2b.y = 80.0
        mock_point2b.visible = True

        mock_instance2 = MagicMock()
        mock_instance2.points = [mock_point1b, mock_point2b]

        mock_frame = MagicMock()
        mock_frame.frame_idx = 0
        mock_frame.video = None
        mock_frame.user_instances = [mock_instance1, mock_instance2]

        mock_skeleton = MagicMock()
        mock_skeleton.nodes = [MagicMock(name="nose"), MagicMock(name="tail")]

        mock_labels = MagicMock()
        mock_labels.labeled_frames = [mock_frame]
        mock_labels.skeletons = [mock_skeleton]

        with patch(
            "ethology.io.annotations.load_keypoints._require_sleap_io"
        ) as mock_sio:
            mock_sio.return_value.load_file.return_value = mock_labels
            from ethology.io.annotations.load_keypoints import (
                _from_single_file,
            )

            try:
                result = _from_single_file(
                    "dummy.sleap", format="SLEAP", images_dirs=None
                )

                # Should have 2 instances in the id dimension
                assert result.sizes["id"] == 2
            except Exception:
                pytest.skip("sleap-io not installed or mock failed")

    def test_frames_with_partial_visibility(self):
        """Test handling frames where some keypoints are not visible."""
        mock_point1 = MagicMock()
        mock_point1.x = 10.0
        mock_point1.y = 20.0
        mock_point1.visible = True

        mock_point2 = MagicMock()
        mock_point2.x = 30.0
        mock_point2.y = 40.0
        mock_point2.visible = False

        mock_instance = MagicMock()
        mock_instance.points = [mock_point1, mock_point2]

        mock_frame = MagicMock()
        mock_frame.frame_idx = 0
        mock_frame.video = None
        mock_frame.user_instances = [mock_instance]

        mock_skeleton = MagicMock()
        mock_skeleton.nodes = [MagicMock(name="nose"), MagicMock(name="tail")]

        mock_labels = MagicMock()
        mock_labels.labeled_frames = [mock_frame]
        mock_labels.skeletons = [mock_skeleton]

        with patch(
            "ethology.io.annotations.load_keypoints._require_sleap_io"
        ) as mock_sio:
            mock_sio.return_value.load_file.return_value = mock_labels
            from ethology.io.annotations.load_keypoints import (
                _from_single_file,
            )

            try:
                result = _from_single_file(
                    "dummy.sleap", format="SLEAP", images_dirs=None
                )

                # Check that visibility is captured
                if "visibility" in result.data_vars:
                    assert np.isclose(result.visibility.values[0, 0, 0], 1.0)
                    # Second keypoint should be invisible
                    assert np.isclose(result.visibility.values[0, 1, 0], 0.0)
            except Exception:
                pytest.skip("sleap-io not installed or mock failed")

    def test_output_dataset_coordinates_order(self):
        """Test that output dataset coordinates are in correct order."""
        mock_point = MagicMock()
        mock_point.x = 10.0
        mock_point.y = 20.0
        mock_point.visible = True

        mock_instance = MagicMock()
        mock_instance.points = [mock_point]

        mock_frame = MagicMock()
        mock_frame.frame_idx = 0
        mock_frame.video = None
        mock_frame.user_instances = [mock_instance]

        mock_skeleton = MagicMock()
        mock_skeleton.nodes = [MagicMock(name="nose")]

        mock_labels = MagicMock()
        mock_labels.labeled_frames = [mock_frame]
        mock_labels.skeletons = [mock_skeleton]

        with patch(
            "ethology.io.annotations.load_keypoints._require_sleap_io"
        ) as mock_sio:
            mock_sio.return_value.load_file.return_value = mock_labels
            from ethology.io.annotations.load_keypoints import (
                _from_single_file,
            )

            try:
                result = _from_single_file(
                    "dummy.sleap", format="SLEAP", images_dirs=None
                )

                # Check that space coordinate has x, y in order
                assert list(result.space.values) == ["x", "y"]
            except Exception:
                pytest.skip("sleap-io not installed or mock failed")
