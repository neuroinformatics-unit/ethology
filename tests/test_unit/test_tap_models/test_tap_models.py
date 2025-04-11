import os
from unittest.mock import patch

import numpy as np
import pytest
import torch

from ethology.tap_models import (
    LIST_OF_SUPPORTED_TAP_MODELS,
    BaseTrackAnyPoint,
    load_video,
)


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


@pytest.mark.parametrize(
    "n_frames, n_individuals, video_shape, tap_model",
    [
        (50, 4, (3, 224, 224), "cotracker"),
    ],
)
def test_save_video(n_frames, n_individuals, video_shape, tap_model, tmpdir):
    tracker = BaseTrackAnyPoint(model=tap_model)
    video = torch.rand(
        (1, n_frames, *video_shape)
    )  # Mock video tensor shape: Batch, Frames, Channels, Height, Width

    video_path = "fake_video_path.mp4"

    pred_tracks = torch.randint(
        0, video_shape[1], (1, n_frames, n_individuals, 2)
    )
    # Mock predicted tracks tensor shape:
    # (batch, frames, individuals, 2D points)

    pred_visibility = torch.randint(
        0, 2, (1, n_frames, n_individuals), dtype=torch.bool
    )  # Mock predicted visibility tensor shape: (batch, frames, individuals)

    tracker.save_video(
        video, video_path, tmpdir, pred_tracks, pred_visibility, 30
    )

    loaded_video = load_video(
        os.path.join(tmpdir, f"{tap_model}_{video_path}")
    )
    # Patch: for some reason, the video while saving it should be saved
    # with 3 less frames according to
    # https://github.com/facebookresearch/co-tracker/blob/main/cotracker/utils/visualizer.py#L156
    # however, while loading the video, it is loaded with 6 more frames,
    # probably something in imageio
    assert loaded_video.shape == (1, n_frames + 6, *video_shape)


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
