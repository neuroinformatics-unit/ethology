from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import xarray as xr

from ethology.io.annotations import save_keypoints


# Mock classes to simulate sleap-io objects
class MockVideo:
    def __init__(self, filename):
        self.filename = filename

    @classmethod
    def from_filename(cls, filename):
        return cls(filename)


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
def test_to_file(mock_require):
    mock_sio = MagicMock()
    mock_require.return_value = mock_sio

    # Mock classes on the mocked module
    mock_sio.Node = MagicMock()
    mock_sio.Skeleton = MagicMock()
    mock_sio.LabeledFrame = MagicMock()
    mock_sio.Instance = MagicMock()
    mock_sio.Video = MagicMock(side_effect=lambda x: MockVideo(x))
    mock_sio.Video.from_filename = MagicMock(
        side_effect=lambda x: MockVideo(x)
    )
    mock_sio.Point = MagicMock()
    mock_sio.Labels = MagicMock()

    # Create dummy dataset
    ds = xr.Dataset(
        {
            "position": (
                ("image_id", "space", "keypoint", "id"),
                np.zeros((1, 2, 1, 1)),
            )
        },
        coords={
            "image_id": [0],
            "space": ["x", "y"],
            "keypoint": ["kp1"],
            "id": [0],
        },
    )
    ds.attrs = {
        "map_image_id_to_filename": {0: "vid.mp4"},
        "map_image_id_to_video": {0: "vid.mp4"},
        "map_keypoint_to_str": {0: "kp1"},
    }

    save_keypoints.to_file(ds, "out.slp", "SLEAP")

    mock_sio.save_file.assert_called()


@patch("ethology.io.annotations.save_keypoints._require_sleap_io")
def test_to_file_missing_video_info(mock_require):
    mock_sio = MagicMock()
    mock_require.return_value = mock_sio

    mock_sio.Node = MagicMock()
    mock_sio.Skeleton = MagicMock()
    mock_sio.LabeledFrame = MagicMock()
    mock_sio.Instance = MagicMock()
    mock_sio.Video = MagicMock()
    mock_sio.Point = MagicMock()
    mock_sio.Labels = MagicMock()

    ds = xr.Dataset(
        {
            "position": (
                ("image_id", "space", "keypoint", "id"),
                np.zeros((1, 2, 1, 1)),
            )
        },
        coords={
            "image_id": [0],
            "space": ["x", "y"],
            "keypoint": ["kp1"],
            "id": [0],
        },
    )
    ds.attrs = {}  # Missing maps

    with pytest.raises(ValueError, match="Missing video or filename"):
        save_keypoints.to_file(ds, "out.slp", "SLEAP")


def test_to_file_unsupported_format():
    with pytest.raises(ValueError, match="Unsupported format"):
        save_keypoints.to_file(MagicMock(), "out.slp", "INVALID")
