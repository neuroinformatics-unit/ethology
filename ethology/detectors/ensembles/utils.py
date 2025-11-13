"""Utility functions for reshaping outputs of ensembles of detectors."""
import numpy as np
import xarray as xr


def get_padding_width(array, max_n):
    """Get pad width for array to max_n detections in the first dimension."""
    pad_width = array.ndim * [(0, 0)]
    pad_width[0] = (0, max_n - array.shape[0])  # before, after
    return pad_width


def pad_to_max_first_dimension(list_arrays):
    """Pad arrays to maximum number across all arrays in the first dimension."""
    max_n_detections = max(array.shape[0] for array in list_arrays)
    list_arrays_padded = [
        np.pad(
            arr,
            get_padding_width(arr, max_n_detections),
            mode="constant",
            constant_values=np.nan,
        )
        for arr in list_arrays
    ]
    return list_arrays_padded


def arrays_to_ds_variables(
    bboxes_x1y1x2y2_array: np.ndarray,
    scores_array: np.ndarray,
    labels_array: np.ndarray,
    id_array: np.ndarray | None = None,
) -> dict[str, xr.DataArray]:
    """Convert arrays to dictionary of dataset variables.

    Parameters
    ----------
    bboxes_x1y1x2y2_array: np.ndarray
        Array of bounding box coordinates with shape
        [Nimages, 4, Nmax_detections], in format x1y1x2y2 in units of pixels.
        Nmax_detections is the maximum number of detections per image.
    scores_array: np.ndarray
        Array of shape [Nimages, Nmax_detections]
    labels_array: np.ndarray
        Array of shape [Nimages, Nmax_detections]
    id_array: np.ndarray | None, optional
        Array of shape [Nmax_detections]. If None, will be set to
        range(Nmax_detections).
    """
    n_images = bboxes_x1y1x2y2_array.shape[0]
    n_max_detections = bboxes_x1y1x2y2_array.shape[-1]
    if id_array is None:
        id_array = np.arange(n_max_detections)

    # centroid dataarray (x, y)
    centroid_da = xr.DataArray(
        data=0.5
        * (
            bboxes_x1y1x2y2_array[:, 0:2, :] + bboxes_x1y1x2y2_array[:, 2:4, :]
        ), 
        dims=["image_id", "space", "id"],
        coords={
            "image_id": np.arange(n_images),
            "space": ["x", "y"],
            "id": id_array,
        },
    )

    # shape dataarray (width, height)
    shape_da = xr.DataArray(
        data=(
            bboxes_x1y1x2y2_array[:, 2:4, :] - bboxes_x1y1x2y2_array[:, 0:2, :]
        ),
        dims=["image_id", "space", "id"],
        coords={
            "image_id": np.arange(n_images),
            "space": ["x", "y"],
            "id": id_array,
        },
    )

    # confidence dataarray
    confidence_da = xr.DataArray(
        data=scores_array,
        dims=["image_id", "id"],
        coords={
            "image_id": np.arange(n_images),
            "id": id_array,
        },
    )

    # label dataarray
    label_da = xr.DataArray(
        data=labels_array,
        dims=["image_id", "id"],
        coords={
            "image_id": np.arange(n_images),
            "id": id_array,
        },
    )

    return {
        "position": centroid_da,
        "shape": shape_da,
        "confidence": confidence_da,
        "label": label_da,
    }
