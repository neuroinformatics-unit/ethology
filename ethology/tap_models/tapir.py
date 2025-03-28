"""TAPIR model implementation for Tracking Any Point.

This module provides a concrete implementation of the TapModel
interface for the TAPIR model.
"""

import mediapy as media
import numpy as np
import torch
from track_any_point import TapModel

from tapnet.tapnet.torch import tapir_model
from tapnet.tapnet.utils import transforms, viz_utils
import torch.nn.functional as F

class TAPIRModel(TapModel):
    """TAPIRModel provides the TAPIR-specific tracking implementation."""


    def __init__(self, weight_file: str | None = None):
        """Initialize TAPIRModel with an optional weight file.

        Args:
            weight_file (str or None): Optional path to the model weights.
        """
        # Initialize TAPIR-specific settings
        pass

    def initialize(self, **kwargs):
        """Initialize the TAPIR model.

        This method disables gradient computation and initializes the
        TAPIR model instance.
        """
        # TAPIR-specific initialization if needed.
        self.model = tapir_model.TAPIR(pyramid_level=1)
        self.model.load_state_dict(
            torch.load("tapnet/checkpoints/bootstapir_checkpoint_v2.pt")
        )
        self.model = self.model.to(self.get_device())
        self.model = self.model.eval()
        torch.set_grad_enabled(False)

    def track(
        self, video_path: str, query_points: list[list], **kwargs
    ) -> torch.Tensor:
        """Track query points in the video using TAPIR.

        Args:
            video_path (str): Path to the input video.
            query_points (list[list]): List of query points.
            **kwargs: Additional arguments for tracking.

        Returns:
            torch.Tensor: Tensor containing tracked points.
        """
        # TAPIR-specific tracking logic.
        # @title Predict Sparse Point Tracks {form-width: "25%"}
        video = media.read_video(video_path)
        resize_height = 256  # @param {type: "integer"}
        resize_width = 256  # @param {type: "integer"}
        num_points = 50  # @param {type: "integer"}

        frames = media.resize_video(video, (resize_height, resize_width))
        query_points = sample_random_points(
            0, frames.shape[1], frames.shape[2], num_points
        )
        frames = torch.tensor(frames).to(self.device)
        query_points = torch.tensor(query_points).to(self.device)

        tracks, visibles = inference(frames, query_points, self.model)

        tracks = tracks.cpu().detach().numpy()
        visibles = visibles.cpu().detach().numpy()
        # Visualize sparse point tracks
        height, width = video.shape[1:3]
        tracks = transforms.convert_grid_coordinates(
            tracks, (resize_width, resize_height), (width, height)
        )
        video_viz = viz_utils.paint_point_track(video, tracks, visibles)
        media.show_video(video_viz, fps=10)
        pass

    def get_model_name(self) -> str:
        """Return the name of the model.

        Returns:
            str: The model name, "tapir".
        """
        return "tapir"

    def get_device(self):
        """Determine and return the device on which the model runs.

        Returns:
            torch.device: 'cuda' if available, else 'cpu'.
        """
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32

    """
    frames = frames.float()
    frames = frames / 255 * 2 - 1
    return frames


def sample_random_points(frame_max_idx, height, width, num_points):
    """Sample random points with (time, height, width) order."""
    y = np.random.randint(0, height, (num_points, 1))
    x = np.random.randint(0, width, (num_points, 1))
    t = np.random.randint(0, frame_max_idx + 1, (num_points, 1))
    points = np.concatenate((t, y, x), axis=-1).astype(
        np.int32
    )  # [num_points, 3]
    return points


def postprocess_occlusions(occlusions, expected_dist):
    """Post-process occlusions to compute visibility.

    Args:
        occlusions: Occlusion scores.
        expected_dist: Expected distance values.

    Returns:
        Tensor: Boolean tensor indicating visible points.
    """
    visibles = (1 - F.sigmoid(occlusions)) * (
        1 - F.sigmoid(expected_dist)
    ) > 0.5
    return visibles


def inference(frames, query_points, model):
    """Perform inference using the TAPIR model.

    This function preprocesses the frames and performs model inference.

    Args:
        frames: Input frames.
        query_points: List of query points.
        model: The TAPIR model instance.

    Returns:
        The inference result from the model.
    """
    # Preprocess video to match model inputs format
    frames = preprocess_frames(frames)
    query_points = query_points.float()
    frames, query_points = frames[None], query_points[None]

    # Model inference
    outputs = model(frames, query_points)
    tracks, occlusions, expected_dist = (
        outputs["tracks"][0],
        outputs["occlusion"][0],
        outputs["expected_dist"][0],
    )

    # Binarize occlusions
    visibles = postprocess_occlusions(occlusions, expected_dist)
    return tracks, visibles
