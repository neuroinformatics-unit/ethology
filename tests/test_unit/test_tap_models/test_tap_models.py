from unittest.mock import patch

import numpy as np
import pytest
import torch

from ethology.tap_models import LIST_OF_SUPPORTED_TAP_MODELS, BaseTrackAnyPoint


@pytest.mark.parametrize(
    "model",
    [
        ("cotracker"),  # Supported tracker type
        ("unsupported_tracker"),
    ],
)
def test_tap_model_type(model):
    if model in LIST_OF_SUPPORTED_TAP_MODELS:
        tracker = BaseTrackAnyPoint(model=model)
        assert tracker.tracker == "cotracker"
    else:
        with pytest.raises(
            ValueError, match=f"Tracker type {model} not supported yet"
        ):
            BaseTrackAnyPoint(model=model)


@pytest.mark.parametrize(
    "video_path, exception, match",
    [
        (
            "invalid_file.mp4",
            FileNotFoundError,
            "Video source file invalid_file.mp4 not found",
        ),
    ],
)
def test_track_with_invalid_video_path(video_path, exception, match):
    tracker = BaseTrackAnyPoint(model="cotracker")
    with pytest.raises(exception, match=match):
        tracker.track(video_path=video_path, query_points=[[0, 0, 0]])


def test_convert_to_movement_dataset():
    # check before conversion to movement and after the data is same.
    n_frames = 50
    n_individuals = 4

    pred_tracks = torch.randint(0, 500, (1, n_frames, n_individuals, 2))
    # pred_visibility = torch.randint(0, 2, (1, 50, 4, 2), dtype=torch.bool)

    tracker = BaseTrackAnyPoint(model="cotracker")
    ds = tracker.convert_to_movement_dataset(pred_tracks)
    x_positions = ds.position.sel(
        space="x"
    ).values  # Shape: (time, keypoints, individuals)
    y_positions = ds.position.sel(space="y").values

    for ind in range(n_individuals):
        print(f"Individual: {ind}")
        x_coords = np.array(x_positions[:, :, ind]).flatten().tolist()
        y_coords = np.array(y_positions[:, :, ind]).flatten().tolist()

        # comparing size of x and y coordinates
        assert len(x_coords) == len(pred_tracks[0, :, ind, 0])
        assert len(y_coords) == len(pred_tracks[0, :, ind, 1])

        # comparing individual elements
        assert x_coords == pred_tracks[0, :, ind, 0].tolist()
        assert y_coords == pred_tracks[0, :, ind, 1].tolist()


def test_track_with_valid_parameters():
    n_frames = 50
    n_keypoints = 1
    tracker = BaseTrackAnyPoint(model="cotracker")
    mock_video = np.random.rand(n_frames, 224, 224, 3)

    query_points = [
        [0, 400, 350],  # frame_number, x, y
        [0, 600, 500],
        [0, 750, 600],
        [0, 900, 200],
    ]
    n_individuals = len(query_points)

    with (
        patch("os.path.isfile", return_value=True),
        patch(
            "cotracker.utils.visualizer.read_video_from_path",
            return_value=mock_video,
        ),
        patch("torch.from_numpy", return_value=torch.from_numpy(mock_video)),
    ):
        ds = tracker.track(
            video_path="fake_video_path.mp4", query_points=query_points
        )

        # checking individual dimensions
        assert ds.position.shape == (n_frames, 2, n_keypoints, n_individuals)
        assert ds.source_software == "cotracker"
        assert len(ds.individuals) == n_individuals
        assert len(ds.keypoints) == n_keypoints
        assert len(ds.time) == n_frames
        assert len(ds.space) == 2
