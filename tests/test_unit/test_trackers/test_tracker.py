import pytest

from ethology.trackers import BaseTracker


@pytest.mark.parametrize(
    "tracker_type, expected",
    [
        ("cotracker", "cotracker"),  # Supported tracker type
        (
            "unsupported_tracker",
            pytest.raises(
                ValueError,
                match="Tracker type unsupported_tracker not supported yet",
            ),
        ),
    ],
)
def test_init(tracker_type, expected):
    if isinstance(expected, str):
        tracker = BaseTracker(tracker_type=tracker_type)
        assert tracker.tracker == expected
    else:
        with expected:
            BaseTracker(tracker_type=tracker_type)


@pytest.mark.parametrize(
    "video_source, exception, match",
    [
        (
            "ethology/trackers/cotracker/assets/invalid_file.mp4",
            FileNotFoundError,
            "Video source file ethology/trackers/cotracker/assets/invalid_file.mp4 not found",  # noqa: E501
        ),
    ],
)
def test_track_with_invalid_video_source(video_source, exception, match):
    tracker = BaseTracker(tracker_type="cotracker")
    with pytest.raises(exception, match=match):
        tracker.track(video_source=video_source, query_points=[[0, 0, 0]])
