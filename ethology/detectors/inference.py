"""Inference utilities for detectors."""

import pandas as pd
import torch

from ethology.detectors.utils import (
    concat_detections_ds,
    detections_dict_as_ds,
)


def run_detector_on_dataset(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,  # dataloader instead?
    device: torch.device,
    # store_sparse: bool = False,
) -> dict:
    """Run detection on each sample of a dataset.

    Note that the dataset transforms are applied to the sampled images.
    The output is a dictionary with the detections per image_id as a
    dictionary. The detections dictionary has the following keys:
    - "boxes": tensor of shape [N, 4]
    - "scores": tensor of shape [N]
    - "labels": tensor of shape [N]
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Run detection for each sample in the dataset
    list_detections_ds = []
    list_image_ids = []
    for image, annotations in dataset:
        # Place image tensor on device and add batch dimension
        image = image.to(device)[None]  # [1, C, H, W]

        with torch.no_grad():
            detections = model(image)

        # Format as xarray dataset
        # [0] to select single batch dimension
        detections_ds = detections_dict_as_ds(detections[0])

        # Append to list
        list_detections_ds.append(detections_ds)
        list_image_ids.append(annotations["image_id"])

    # Concatenate all detections datasets along image_id dimension
    detections_dataset = concat_detections_ds(
        list_detections_ds,
        pd.Index(list_image_ids, name="image_id"),
    )  # [image_id, model, annot_id]

    # Add image_width and image_height as attributes
    # (we assume all images in the dataset have the same width and height
    # as the last image)
    detections_dataset.attrs["image_width"] = image.shape[-1]  # columns
    detections_dataset.attrs["image_height"] = image.shape[-2]  # rows

    return detections_dataset


def run_detector_on_dataloader(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> dict:
    """Run detection on a dataloader.

    The output is a dictionary with the detections per batch as a list.
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
        image_batch = [img.to(device) for img in image_batch]  # [B, C, H, W]

        # Run detection
        with torch.no_grad():
            detections_batch = model(
                image_batch
            )  # list of n-batch dictionaries

        # Add to dict
        detections_per_batch[batch_idx] = detections_batch

    # # Format as xarray dataset
    # detections_dataset = _detections_per_image_id_as_ds(
    #     detections_per_image_id
    # )

    return detections_per_batch


# def collate_fn_varying_n_bboxes(batch: tuple) -> tuple:
#     """Collate function for dataloader with varying number of bounding boxes.

#     A custom function is needed for detection
#     because the number of bounding boxes varies
#     between images of the same batch.
#     See https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html#data-loading-and-training-loop

#     Parameters
#     ----------
#     batch : tuple
#         a tuple of 2 tuples, the first one holding all images in the batch,
#         and the second one holding the corresponding annotations.

#     Returns
#     -------
#     tuple
#         a tuple of length = batch size, made up of (image, annotations)
#         tuples.

#     """
#     return tuple(zip(*batch, strict=False))


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
