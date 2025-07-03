"""Inference utilities for detectors."""

import numpy as np
import torch


def run_detector_on_dataset(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,  # dataloader instead?
    device: torch.device,
) -> dict:
    """Run detection on each sample of a dataset.

    Note that the dataset transforms are applied to the sampled images.
    The output is a dictionary with the detections per sample as a dictionary.
    The detections dictionary has the following keys:
    - "boxes": tensor of shape [N, 4]
    - "scores": tensor of shape [N]
    - "labels": tensor of shape [N]
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Run detection
    detections_per_sample = {}
    for idx, (image, annotations) in enumerate(dataset):
        # Place image tensor on device and add batch dimension
        image = image.to(device)[None]  # [1, C, H, W]

        # Run detection
        with torch.no_grad():
            detections = model(image)[0]  # select single batch dimension

        # Add to dict
        detections_per_sample[idx] = detections

    return detections_per_sample


def run_detector_on_dataloader(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run detection on a dataloader.

    The output is a list of dictionary with the detections per batch.
    The detections dictionary has the following keys:
    - "boxes": tensor of shape [N, 4]
    - "scores": tensor of shape [N]
    - "labels": tensor of shape [N]
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Compute detections per batch
    detections_per_batch = {}
    for batch_idx, (image_batch, _annotations_batch) in enumerate(dataloader):
        # Place batch of images on device
        image_batch = image_batch.to(device)  # [B, C, H, W]

        # Run detection
        with torch.no_grad():
            detections_batch = model(
                image_batch
            )  # list of n-batch dictionaries

        # Add to dict
        detections_per_batch[batch_idx] = detections_batch

    return detections_per_batch


# def run_detector_on_image(
#     model: torch.nn.Module,
#     image: torch.Tensor,
#     device: torch.device,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Run detection on an image."""
#     pass


# def run_detector_on_video(
#     model: torch.nn.Module,
#     video_path: str,
#     device: torch.device,
# ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#     """Run detection on a video."""
#     pass
