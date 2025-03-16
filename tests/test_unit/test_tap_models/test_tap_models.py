from unittest.mock import patch

import numpy as np
import pytest
import torch

from ethology.tap_models import LIST_OF_SUPPORTED_TAP_MODELS, BaseTrackAnyPoint


@pytest.mark.parametrize(
    "tracker_type",
    [
        ("cotracker"),  # Supported tracker type
        ("unsupported_tracker"),
    ],
)
def test_init(tracker_type):
    if tracker_type in LIST_OF_SUPPORTED_TAP_MODELS:
        tracker = BaseTrackAnyPoint(tracker_type=tracker_type)
        assert tracker.tracker == "cotracker"
    else:
        with pytest.raises(
            ValueError, match=f"Tracker type {tracker_type} not supported yet"
        ):
            BaseTrackAnyPoint(tracker_type=tracker_type)


@pytest.mark.parametrize(
    "video_path, exception, match",
    [
        (
            "ethology/trackers/cotracker/assets/invalid_file.mp4",
            FileNotFoundError,
            "Video source file ethology/trackers/cotracker/assets/invalid_file.mp4 not found",  # noqa: E501
        ),
    ],
)
def test_track_with_invalid_video_path(video_path, exception, match):
    tracker = BaseTrackAnyPoint(tracker_type="cotracker")
    with pytest.raises(exception, match=match):
        tracker.track(video_path=video_path, query_points=[[0, 0, 0]])


def test_track_with_valid_parameters():
    tracker = BaseTrackAnyPoint(tracker_type="cotracker")
    mock_video = np.random.rand(50, 224, 224, 3)

    query_points = [
        [0, 400, 350],  # frame_number, x, y
        [0, 600, 500],
        [0, 750, 600],
        [0, 900, 200],
    ]

    with (
        patch("os.path.isfile", return_value=True),
        patch(
            "cotracker.utils.visualizer.read_video_from_path",
            return_value=mock_video,
        ),
        patch("torch.from_numpy", return_value=torch.from_numpy(mock_video)),
    ):
        path = tracker.track(
            video_path="fake_video_path.mp4", query_points=query_points
        )
        assert list(path.shape) == [1, 50, 4, 2]
