"""Module for tracking any point using various tap models.

This module defines the abstract base class and the concrete
implementation for the CoTracker model.
"""
import os
from abc import ABC, abstractmethod

import torch
from cotracker.predictor import CoTrackerPredictor
from cotracker.utils.visualizer import Visualizer, read_video_from_path
from movement.io.load_poses import from_numpy

# List of supported models (extend this list as new models are added)
LIST_OF_SUPPORTED_TAP_MODELS = ["cotracker", "tapir"]


class TapModel(ABC):
    """Abstract base class for all tap models used in Tracking Any Point."""

    @abstractmethod
    def initialize(self, **kwargs):
        """Optional method to initialize the model if needed."""
        pass

    @abstractmethod
    def track(
        self, video_path: str, query_points: list[list], **kwargs
    ) -> torch.Tensor:
        """Track the query points in the provided video.

        Parameters
        ----------
            video_path (str): Path to the input video file.
            query_points (list[list]): 2D list of query points.

        Returns
        -------
            torch.Tensor: Tensor containing the tracks.

        """
        pass

    def convert_to_movement_dataset(self, pred_tracks: torch.Tensor):
        """Convert the predicted tracks to a movement dataset."""
        # pred_tracks.shape =  [batch, frames, query_points, 2d points]
        # Adjust shape to: [frames, 2d points, 1 keypoint per query point, no. of query points]
        pred_tracks = pred_tracks.cpu().numpy().squeeze(0)
        ds = from_numpy(
            position_array=pred_tracks.transpose(0, 2, 1)[:, :, None, :],
            source_software=self.get_model_name(),
        )
        return ds

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the name of the model as a string."""
        pass


class CoTrackerModel(TapModel):
    """Concrete implementation for the CoTracker model."""

    def __init__(self, weight_file: str | None = None):
        # Validate the weight file if provided.
        if weight_file and not os.path.isfile(weight_file):
            raise FileNotFoundError(
                f"Tracker model weights {weight_file} not found"
            )

        # Initialize the model
        if weight_file is not None:
            self.model = CoTrackerPredictor(checkpoint=weight_file)
        else:
            self.model = torch.hub.load(
                "facebookresearch/co-tracker", "cotracker3_offline"
            )

    def initialize(self, **kwargs):
        # If the CoTracker model needed additional initialization, it would go here.
        pass

    def track(
        self,
        video_path: str,
        query_points: list[list],
        save_dir: str = "processed_videos",
        save_results: bool = False,
    ) -> torch.Tensor:
        # Check if the video source file exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(
                f"Video source file {video_path} not found"
            )

        # Load video and process shape: (Frames, Height, Width, Channels) -> (Batch, Frames, Channels, Height, Width)
        video = read_video_from_path(video_path)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)[None].float()

        # Convert query points to tensor
        query_tensor = torch.tensor(query_points, dtype=torch.float32)

        # Set device
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        video = video.to(device)
        query_tensor = query_tensor.to(device)

        # Run the model's tracking method
        pred_tracks, pred_visibility = self.model(
            video, queries=query_tensor[None]
        )

        # Optionally, save the visualized results
        if save_results:
            os.makedirs(save_dir, exist_ok=True)
            file_name = f"cotracker_{os.path.splitext(os.path.basename(video_path))[0]}"
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

        return self.convert_to_movement_dataset(pred_tracks)

    def get_model_name(self) -> str:
        return "cotracker"


class TapModelFactory:
    """Factory class to create TapModel instances."""

    @staticmethod
    def create_model(
        model_type: str, weight_file: str | None = None
    ) -> TapModel:
        model_type = model_type.lower()
        if model_type == "cotracker":
            return CoTrackerModel(weight_file=weight_file)
        elif model_type == "tapir":
            # Placeholder for when a TAPIR implementation is available
            raise NotImplementedError("TAPIR model is not implemented yet.")
        else:
            raise ValueError(f"Unknown model type: {model_type}")
