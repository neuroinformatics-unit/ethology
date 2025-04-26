"""Tracking Any Point module."""

import os
from typing import Literal

import torch
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import (
    Visualizer,
    read_video_from_path,
)
from loguru import logger
from movement.io.load_poses import from_numpy
from movement.validators.datasets import ValidPosesDataset

LIST_OF_SUPPORTED_TAP_MODELS = ["cotracker"]


def get_device():
    """Get the device to run the model on.

    priority: cuda > mps > cpu

    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    return device


def load_video(video_path):
    """Load the video from the given path.

    Parameters
    ----------
    video_path: str
        Path to the video file.

    Returns
    -------
    torch.Tensor
        Tensor containing the video frames of shape
        (Batch, Frames, Channels, Height, Width).

    """
    video = read_video_from_path(video_path)

    # (Frames, Height, Width, Channels) ->
    # (Batch, Frames, Channels, Height, Width)
    video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()
    return video


class BaseTrackAnyPoint:
    """Base class for all Tracking Any Point trackers.

    Parameters
    ----------
    model: str
        Type of Point tracker, current implementation supports ["cotracker"]
    checkpoint_path: str
        Path to the model weight of the tracker selected

    """

    def __init__(
        self,
        model: Literal["cotracker"],
        checkpoint_path: str | None = None,
    ):
        """Initialize the tracker."""
        self.tracker = model

        if model not in LIST_OF_SUPPORTED_TAP_MODELS:
            raise ValueError(f"Tracker type {model} not supported yet")

        # load model
        if model == "cotracker":
            if checkpoint_path and not os.path.isfile(checkpoint_path):
                raise FileNotFoundError(
                    f"Tracker model weights {checkpoint_path} not found"
                )

            if checkpoint_path is not None:
                self.model = CoTrackerPredictor(checkpoint=checkpoint_path)
            else:
                self.model = torch.hub.load(
                    "facebookresearch/co-tracker", "cotracker3_offline"
                )

    def convert_to_movement_dataset(
        self, pred_tracks, individuals
    ) -> ValidPosesDataset:
        """Convert the predicted tracks to movement dataset.

        Parameters
        ----------
        pred_tracks: torch.Tensor
            Tensor containing tracks of each query
            point across the frames of size
            [batch, frame, query_points, X, Y].
        individuals: list
            List of names of individuals tracked.

        Returns
        -------
        movement.ValidPosesDataset
            Dataset containing the predicted tracks.

        """
        # pred_tracks.shape =
        # [batch, frames,
        # keypoints per individual * number of individuals, 2d points]
        # goes into positional array with shape
        # [frames, 2d points, keypoints, individuals]
        num_individuals = len(individuals)
        pred_tracks = pred_tracks.cpu().numpy().squeeze(0)
        # [frames, keypoints per individual * number of individuals, 2d points]
        num_keypoints = pred_tracks.shape[1] // num_individuals
        pred_tracks = pred_tracks.reshape(
            pred_tracks.shape[0], num_individuals, num_keypoints, 2
        )  # [frames, individuals, keypoints, 2d points]

        ds = from_numpy(
            position_array=pred_tracks.transpose(
                0, 3, 2, 1
            ),  # [frames, 2d points, keypoints, individuals]
            individual_names=individuals,
            source_software="cotracker",
        )
        return ds

    def save_video(
        self,
        video: torch.Tensor,
        video_path: str,
        save_dir: str,
        pred_tracks: torch.Tensor,
        pred_visibility: torch.Tensor,
        fps: int,
    ):
        """Save the processed video.

        Parameters
        ----------
        video: torch.Tensor
            Tensor containing video frames.
        video_path: str
            Path to the input video file to extract
            name from.
        save_dir: str
            Directory path to save the processed video.
        pred_tracks: torch.Tensor
            Tensor containing tracks of each query
            point across the frames of size
            [batch, frame, query_points, X, Y].
        pred_visibility: torch.Tensor
            Tensor containing visibility of each query
            point across the frames of size
            [batch, frame, query_points].
        fps: int
            Frames per second to save the video at.

        """
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        file_name = f'{self.tracker}_{video_path.split("/")[-1].split(".")[0]}'

        vis = Visualizer(
            save_dir=save_dir,
            linewidth=6,
            mode="cool",
            tracks_leave_trace=-1,
            fps=fps,
        )
        vis.visualize(
            video=video,
            tracks=pred_tracks,
            visibility=pred_visibility,
            filename=file_name,
        )

    def track(
        self,
        video_path: str,
        query_points: dict[str, list],
        save_dir: str | None = None,
        fps: int = 10,
    ) -> ValidPosesDataset:
        """Track the query points in the video source.

        Parameters
        ----------
        video_path: str
            Path to the input video file.
        query_points: dict
            Dictionary of query points for each individual,
            key is the individual id and value is a
            2D list of shape (Q,3) where Q is the
            number of query points and each Q containing
            Frame-Number to start tracking from, X, Y
        save_dir: str
            If given, will save the processed video
            in the given directory.
        fps: int
            Frames per second to save the video at.
            Default is 10.

        Returns
        -------
        movement.validators.datasets.ValidPosesDataset
            Dataset containing the predicted tracks. Current implementation
            supports 1 keypoint per individual.

        """
        # check if video source file exists in path
        if not os.path.isfile(video_path):
            raise FileNotFoundError(
                f"Video source file {video_path} not found"
            )

        # load video
        video_data = load_video(video_path)

        # dict to tensor
        query_tensor = torch.tensor(
            list(query_points.values()), dtype=torch.float32
        )

        # set device
        device = get_device()
        logger.info(f"Running inference on {device}...")

        # handle out of memory errors
        try:
            self.model = self.model.to(device)
            video_data = video_data.to(device)
            query_tensor = query_tensor.to(device)
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.error("Out of memory error.")
                logger.debug("Switching to CPU...")
                torch.cuda.empty_cache()

                self.model = self.model.to("cpu")
                video_data = video_data.to("cpu")
                query_tensor = query_tensor.to("cpu")
            else:
                raise e

        # Run tap model
        logger.info(f"Starting Tracking Any Point using {self.tracker}...")

        all_keypoint_tracks, all_keypoint_visibility = (
            torch.tensor([]),
            torch.tensor([]),
        )
        for query in query_tensor:
            pred_tracks, pred_visibility = self.model(
                video_data, queries=query[None]
            )
            all_keypoint_tracks = torch.cat(
                (all_keypoint_tracks, pred_tracks.to("cpu")), dim=2
            )
            all_keypoint_visibility = torch.cat(
                (all_keypoint_visibility, pred_visibility.to("cpu")), dim=2
            )

        logger.info("tracking complete")

        if save_dir:
            self.save_video(
                video_data,
                video_path,
                save_dir,
                all_keypoint_tracks,
                all_keypoint_visibility,
                fps,
            )
            logger.success("Video saved successfully!")

        ds = self.convert_to_movement_dataset(
            all_keypoint_tracks, query_points.keys()
        )
        logger.success("Tracking Any Point completed successfully!")

        return ds
