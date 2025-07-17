"""Inference utilities for detectors."""

import numpy as np
import torch
import xarray as xr


def _pad_sequence_along_detections_dim(
    array: np.ndarray, max_n_detections_per_image: int
) -> tuple:
    """Return sequence for padding input array along detections dimension."""
    pad_detections_per_image = max_n_detections_per_image - array.shape[0]
    return tuple(
        (0, pad_detections_per_image) if i == 0 else (0, 0)
        for i in range(array.ndim)
    )


def _detections_per_image_id_as_ds(
    detections_per_image_id: dict,
) -> xr.Dataset:
    """Reshape detections per sample as xarray dataset."""
    # Get coordinates
    list_image_id_coords = list(detections_per_image_id.keys())
    list_space_coords = ["x", "y"]
    max_n_detections_per_image = max(
        [
            detections["boxes"].shape[0]
            for detections in detections_per_image_id.values()
        ]
    )

    list_id_coords = list(range(max_n_detections_per_image))  # per frame
    coords_dict = {
        "image_id": list_image_id_coords,
        "space": list_space_coords,
        "id": list_id_coords,  # per frame
    }
    coords_dict_no_space = coords_dict.copy()
    del coords_dict_no_space["space"]

    # Get lists of data arrays
    list_centroid_arrays = [
        (
            detections["boxes"].cpu().numpy()[:, 0:2]
            + detections["boxes"].cpu().numpy()[:, 2:4]
        )
        * 0.5
        for detections in detections_per_image_id.values()
    ]

    list_shape_arrays = [
        detections["boxes"].cpu().numpy()[:, 2:4]
        - detections["boxes"].cpu().numpy()[:, 0:2]
        for detections in detections_per_image_id.values()
    ]

    list_confidence_arrays = [
        detections["scores"].cpu().numpy()  # .reshape(-1, 1)
        for detections in detections_per_image_id.values()
    ]

    list_label_arrays = [
        detections["labels"].cpu().numpy()  # .reshape(-1, 1)
        for detections in detections_per_image_id.values()
    ]

    # Define arrays to create
    arrays_dict = {
        "centroids": {  # --> change to position
            "data": list_centroid_arrays,
            "coords": coords_dict,
            "pad_value": np.nan,
        },
        "shape": {
            "data": list_shape_arrays,
            "coords": coords_dict,
            "pad_value": np.nan,
        },
        "confidence": {
            "data": list_confidence_arrays,
            "coords": coords_dict_no_space,
            "pad_value": np.nan,
        },
        "label": {
            "data": list_label_arrays,
            "coords": coords_dict_no_space,
            "pad_value": -1,
        },
    }

    # Create all DataArrays in a loop
    data_arrays = {}
    for name in arrays_dict:
        data_arrays[name] = xr.DataArray(
            data=np.stack(
                [
                    np.pad(
                        array,
                        _pad_sequence_along_detections_dim(
                            array, max_n_detections_per_image
                        ),
                        mode="constant",
                        constant_values=arrays_dict[name]["pad_value"],
                    ).T
                    for array in arrays_dict[name]["data"]
                ],
                axis=0,  # need to pad with nans for constant shape
            ),
            dims=list(arrays_dict[name]["coords"].keys()),
            coords=arrays_dict[name]["coords"],
        )

    return xr.Dataset(data_vars=data_arrays)


def run_detector_on_dataset(
    model: torch.nn.Module,
    dataset: torch.utils.data.Dataset,  # dataloader instead?
    device: torch.device,
) -> dict:
    """Run detection on each sample of a dataset.

    Note that the dataset transforms are applied to the sampled images.
    The output is a dictionary with the detections per image_id as a dictionary.
    The detections dictionary has the following keys:
    - "boxes": tensor of shape [N, 4]
    - "scores": tensor of shape [N]
    - "labels": tensor of shape [N]
    """
    # Ensure model is in evaluation mode
    model.eval()

    # Run detection
    detections_per_image_id = {}
    for image, annotations in dataset:
        # Place image tensor on device and add batch dimension
        image = image.to(device)[None]  # [1, C, H, W]

        # Run detection
        with torch.no_grad():
            detections = model(image)[0]  # select single batch dimension

        # Add to dict with key = image_id
        detections_per_image_id[annotations["image_id"]] = detections

    # Format as xarray dataset
    detections_dataset = _detections_per_image_id_as_ds(
        detections_per_image_id
    )

    return detections_dataset


def _detections_per_batch_as_ds(
    detections_per_batch: dict,
) -> xr.Dataset:
    """Reshape detections per batch as xarray dataset."""
    pass


def run_detector_on_dataloader(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    return detections_per_batch


def collate_fn_varying_n_bboxes(batch: tuple) -> tuple:
    """Collate function for dataloader with varying number of bounding boxes.

    A custom function is needed for detection
    because the number of bounding boxes varies
    between images of the same batch.
    See https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html#data-loading-and-training-loop

    Parameters
    ----------
    batch : tuple
        a tuple of 2 tuples, the first one holding all images in the batch,
        and the second one holding the corresponding annotations.

    Returns
    -------
    tuple
        a tuple of length = batch size, made up of (image, annotations)
        tuples.

    """
    return tuple(zip(*batch, strict=False))


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
