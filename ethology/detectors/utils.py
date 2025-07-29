"""Utility functions for transforming detection datasets."""

import numpy as np
import pandas as pd
import torch
import xarray as xr


def concat_detections_ds(
    list_detections_ds: list[xr.Dataset], index: pd.Index
) -> xr.Dataset:
    """Concatenate detections datasets along new dimension."""
    # Check index has name
    if index.name is None:
        raise ValueError("Index must have a name")

    # Concatenate along new dimension
    ds = xr.concat(
        list_detections_ds,
        index,
    )

    # ensure "label" array is padded with -1 rather than nan
    if "label" in ds.data_vars:
        ds["label"] = ds.label.fillna(-1).astype(int)

    return ds


def detections_dict_as_ds_batch(
    list_detections: list[dict],
) -> list[xr.Dataset]:
    """Reshape list of detections dictionaries as xarray dataset.

    Input is list of detections dictionaries with keys:
    - "boxes": tensor of shape [N, 4], x1y1x2y2 in pixels
    - "scores": tensor of shape [N]
    - "labels": tensor of shape [N]

    Output is a list of xarray datasets, one for each image in the batch.
    """
    return [
        detections_dict_as_ds(detections) for detections in list_detections
    ]


def detections_dict_as_ds(detections: dict) -> xr.Dataset:
    """Reshape detections dictionaryas xarray dataset.

    Input is detections dictionary with keys:
    - "boxes": tensor of shape [N, 4], x1y1x2y2 in pixels
    - "scores": tensor of shape [N]
    - "labels": tensor of shape [N]

    Output is xarray dataset with keys:
    - "position": xarray.DataArray of shape [2, N] (space, annot_id)
    - "shape": xarray.DataArray of shape [2, N] (space, annot_id)
    - "confidence": xarray.DataArray of shape [N] (annot_id)
    - "label": xarray.DataArray of shape [N] (annot_id)
    """
    # Place tensors on cpu if required & convert to numpy array
    detections = {
        key: value.cpu().numpy() if isinstance(value, torch.Tensor) else value
        for key, value in detections.items()
    }

    return detections_x1y1_x2y2_as_ds(
        detections["boxes"],
        detections["scores"],
        detections["labels"],
    )


def detections_x1y1_x2y2_as_da_tuple(
    x1y1_x2y2_array: np.ndarray,
    scores_array: np.ndarray,
    labels_array: np.ndarray,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Reshape detections array as xarray dataset.

    Input is detections array with shape [N, 4], x1y1x2y2 in pixels
    """
    # Create xarray dataset
    n_detections = x1y1_x2y2_array.shape[0]
    centroid_da = xr.DataArray(
        data=0.5
        * (
            x1y1_x2y2_array[:, 0:2] + x1y1_x2y2_array[:, 2:4]
        ).T,  # space, annot ID
        dims=["space", "id"],
        coords={
            "space": ["x", "y"],
            "id": list(range(n_detections)),
        },
    )

    shape_da = xr.DataArray(
        data=(
            x1y1_x2y2_array[:, 2:4] - x1y1_x2y2_array[:, 0:2]
        ).T,  # space, annot ID
        dims=["space", "id"],
        coords={
            "space": ["x", "y"],
            "id": list(range(n_detections)),
        },
    )

    confidence_da = xr.DataArray(
        data=scores_array,
        dims=["id"],
        coords={"id": list(range(n_detections))},
    )

    label_da = xr.DataArray(
        data=labels_array,
        dims=["id"],
        coords={"id": list(range(n_detections))},
    )

    return centroid_da, shape_da, confidence_da, label_da


def detections_x1y1_x2y2_as_ds(
    x1y1_x2y2_array: np.ndarray,
    scores_array: np.ndarray,
    labels_array: np.ndarray,
) -> xr.Dataset:
    """Reshape detections array as xarray dataset.

    Input is detections array with shape [N, 4], x1y1x2y2 in pixels
    """
    # Remove nan rows
    slc_nan_rows = np.any(np.isnan(x1y1_x2y2_array), axis=1)
    x1y1_x2y2_array = x1y1_x2y2_array[~slc_nan_rows]
    scores_array = scores_array[~slc_nan_rows]
    labels_array = labels_array[~slc_nan_rows]

    # Create dataarrays for dataset
    centroid_da, shape_da, confidence_da, label_da = (
        detections_x1y1_x2y2_as_da_tuple(
            x1y1_x2y2_array, scores_array, labels_array
        )
    )

    return xr.Dataset(
        data_vars={
            "position": centroid_da,
            "shape": shape_da,
            "confidence": confidence_da,
            "label": label_da,
        }
    )


def add_bboxes_min_max_corners(ds):
    """Add xy_min and xy_max arrays to ds.

    # Compare to box_convert in testing?
    box_convert(
        torch.from_numpy(np.c_[ds.position.T, ds.shape.T]),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    )
    """
    ds["xy_min"] = ds.position - 0.5 * ds.shape
    ds["xy_max"] = ds.position + 0.5 * ds.shape
    return ds
