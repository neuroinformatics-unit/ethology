"""Tracking Any Point module."""

import os
from typing import Literal

import torch
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import (
    Visualizer,
    read_video_from_path,
)

LIST_OF_SUPPORTED_TAP_MODELS = ["cotracker"]


class BaseTrackAnyPoint:
    """Base class for all Tracking Any Point trackers.

    Parameters
    ----------
    tracker_type: str
        Type of Point tracker, current implementation supports ["cotracker"]
    weight_file: str
        Path to the model weight of the tracker selected

    """

    def __init__(
        self,
        tracker_type: Literal["cotracker"],
        weight_file: str | None = None,
    ):
        """Initialize the tracker."""
        self.tracker = tracker_type

        if tracker_type not in LIST_OF_SUPPORTED_TAP_MODELS:
            raise ValueError(f"Tracker type {tracker_type} not supported yet")

        # load model
        if tracker_type == "cotracker":
            if weight_file and not os.path.isfile(weight_file):
                raise FileNotFoundError(
                    f"Tracker model weights {weight_file} not found"
                )

            if weight_file is not None:
                self.model = CoTrackerPredictor(checkpoint=weight_file)
            else:
                self.model = torch.hub.load(
                    "facebookresearch/co-tracker", "cotracker3_offline"
                )

    def track(
        self,
        video_path: str,
        query_points: list[list],
        save_dir: str = "processed_videos",
        save_results: bool = False,
    ) -> torch.Tensor:
        """Track the query points in the video source.

        Parameters
        ----------
        video_path: str
            Path to the input video file.
        query_points: list
            2D List of shape (Q,3) where Q is the
            number of query points and each Q containing
            X, Y, Frame-Number to start tracking from
        save_dir: str
            Directory path to save the processed video
        save_results: bool
            Flag to save the results as a video file

        Returns
        -------
        torch.Tensor
            Tensor containing tracks of each query
            point across the frames of size
            [batch, frame, query_points, X, Y].
            Example: 50 frames video with 4 query_points
            will give a tensor of shape [1, 50, 4, 2]

        """
        # check if video source file exists in path
        if not os.path.isfile(video_path):
            raise FileNotFoundError(
                f"Video source file {video_path} not found"
            )
        else:
            print(f"Video source file {video_path} found")

        # load video
        video = read_video_from_path(video_path)

        # (Frames, Height, Width, Channels) ->
        # (Batch, Frames, Channels, Height, Width)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

        # List to tensor
        query_tensor = torch.tensor(query_points, dtype=torch.float32)

        # set device
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        self.model = self.model.to(device)
        video = video.to(device)
        query_tensor = query_tensor.to(device)
        print("video shape", video.shape)
        pred_tracks, pred_visibility = self.model(
            video, queries=query_tensor[None]
        )

        if save_results:
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            file_name = (
                f'{self.tracker}_{video_path.split("/")[-1].split(".")[0]}'
            )

            vis = Visualizer(
                save_dir=save_dir,
                linewidth=6,
                mode="cool",
                tracks_leave_trace=-1,
            )
            vis.visualize(
                video=video,
                tracks=pred_tracks,
                visibility=pred_visibility,
                filename=file_name,
            )

        return pred_tracks
