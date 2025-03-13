"""Trackers module."""

import os
from typing import Literal

import torch

from .cotracker.cotracker.utils.visualizer import (
    Visualizer,
    read_video_from_path,
)


class BaseTracker:
    """Base class for all trackers.

    Args:
     video_source: str
     query_points: list
     tracker: str

    """

    def __init__(self, tracker_type: Literal["cotracker"]):
        """Initialize the tracker."""
        if tracker_type not in ["cotracker"]:
            raise ValueError(f"Tracker type {tracker_type} not supported yet")

        self.tracker = tracker_type

    def track(
        self,
        video_source: str,
        query_points: list[list],
        save_results: bool = False,
    ) -> torch.Tensor:
        """Track the query points in the video source.

        Args:
         video_source: str
         query_points: list
         save_results: bool
        Returns:
         path of each point cross the frames [Batch, Frames, X, Y].

        """
        # check if video source file exists in path
        if not os.path.isfile(video_source):
            raise FileNotFoundError(
                f"Video source file {video_source} not found"
            )
        else:
            print(f"Video source file {video_source} found")

        # load video
        video = read_video_from_path(video_source)

        # (Frames, Height, Width, Channels) ->
        # (Batch, Frames, Channels, Height, Width)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

        # load model
        if self.tracker == "cotracker":
            model = torch.hub.load(
                "facebookresearch/co-tracker", "cotracker3_offline"
            )

        # List to tensor
        query_tensor = torch.tensor(query_points, dtype=torch.float32)

        # set device
        device = (
            torch.device("cuda")
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

        model = model.to(device)
        video = video.to(device)
        query_tensor = query_tensor.to(device)

        pred_tracks, pred_visibility = model(video, queries=query_tensor[None])

        if save_results:
            vis = Visualizer(
                save_dir="../ethology/video",
                linewidth=6,
                mode="cool",
                tracks_leave_trace=-1,
            )
            vis.visualize(
                video=video,
                tracks=pred_tracks,
                visibility=pred_visibility,
                filename=f'{self.tracker}_{video_source.split("/")[-1].split(".")[0]}',
            )

        return pred_tracks
