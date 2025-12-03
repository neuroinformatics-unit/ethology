"""Wrappers around ensemble-boxes fusion functions."""

from collections.abc import Callable
from functools import partial
from typing import Literal, TypedDict, Unpack

import ensemble_boxes
import numpy as np
import xarray as xr

from ethology.validators.detections import (
    ValidBboxDetectionsDataset,
    ValidBboxDetectionsEnsembleDataset,
)
from ethology.validators.utils import _check_input, _check_output

VALID_FUSION_METHODS = {
    "weighted_boxes_fusion": ensemble_boxes.weighted_boxes_fusion,
    "nms": ensemble_boxes.nms,
    "soft_nms": ensemble_boxes.soft_nms,
    "non_maxium_weighted": ensemble_boxes.non_maximum_weighted,
}

fusion_method_type = Literal[
    "weighted_boxes_fusion", "nms", "soft_nms", "non_maxium_weighted"
]


class _TypeFusionKwargs(TypedDict, total=False):
    """Type hints for fusion method kwargs.

    Parameters for methods as described in the ensemble_boxes documentation.
    See https://github.com/ZFTurbo/Weighted-Boxes-Fusion


    Parameters
    ----------
    weights: list[float]
        Weights for each model.
    iou_thr: float
        IoU threshold for detections to be considered a true positive
        IoU threshold for detections to be considered a true positive
        during fusion.
    skip_box_thr: float
        Exclude from fusion boxes with confidence below this value.
    sigma: float
        Sigma for soft NMS.
    thresh: float
        Threshold for boxes to keep after soft NMS.
    conf_type: Literal["avg", "box_and_model_avg", "absent_model_aware_avg"]
        Method to compute the confidence score of the fused detections.

        - "avg": Average confidence score of the fused detections (default).
        - 'box_and_model_avg': box and model wise hybrid weighted average.
        - 'absent_model_aware_avg': weighted average that takes into account
          the absent model.
    allows_overflow: bool
        Whether to allow the confidence score of the fused detections to
        exceed 1.

    """

    weights: list[float] | None
    iou_thr: float
    skip_box_thr: float
    sigma: float
    thresh: float
    conf_type: Literal["avg", "box_and_model_avg", "absent_model_aware_avg"]
    allows_overflow: bool


@_check_input(ValidBboxDetectionsEnsembleDataset)
@_check_output(ValidBboxDetectionsDataset)
def fuse_detections(
    ensemble_detections_ds: xr.Dataset,
    fusion_method: fusion_method_type,
    fusion_method_kwargs: dict | None = None,
    max_n_detections: int | None = None,
) -> xr.Dataset:
    """Fuse ensemble detections across models using WBF.

    You can set a max_n_detections if upper bound is known a prior to
    reduce memory usage.

    """
    # Check if image_width_height defined in dataset
    image_shape = ensemble_detections_ds.attrs.get("image_shape")
    if image_shape is None:
        raise KeyError(
            "Required attribute 'image_shape' not found in the dataset "
            "attributes. Please ensure the dataset has 'image_shape' "
            "(width, height in pixels) in its attributes."
        )
    else:
        image_width_height = _validate_image_shape(image_shape)

    # Compute upper bound of max_n_detections
    if not max_n_detections:
        max_n_detections = _estimate_max_n_detections(ensemble_detections_ds)

    # Build single-image partial fusion function for the selected method
    if fusion_method not in VALID_FUSION_METHODS:
        raise ValueError(
            f"Invalid fusion method: {fusion_method}. "
            f"Valid methods are: {list(VALID_FUSION_METHODS.keys())}"
        )
    fusion_function = VALID_FUSION_METHODS[fusion_method]
    _fuse_single_image_detections_partial = partial(
        _fuse_single_image_detections, fusion_function
    )

    # Run fusion across image_id using apply_ufunc
    centroid_fused_da, shape_fused_da, confidence_fused_da, label_fused_da = (
        xr.apply_ufunc(
            _fuse_single_image_detections_partial,
            ensemble_detections_ds.position,  # .data array is passed
            ensemble_detections_ds.shape,
            ensemble_detections_ds.confidence,
            ensemble_detections_ds.label,
            kwargs={
                "image_width_height": image_width_height,
                "max_n_detections": max_n_detections,
                **(fusion_method_kwargs if fusion_method_kwargs else {}),
            },
            input_core_dims=[  # do not broadcast across these
                ["space", "id", "model"],  # centroid
                ["space", "id", "model"],  # shape
                ["id", "model"],  # confidence
                ["id", "model"],  # label
            ],
            output_core_dims=[  # do not broadcast across these
                ["space", "id"],  # centroid
                ["space", "id"],  # shape
                ["id"],  # confidence
                ["id"],  # label
            ],
            vectorize=True,
            # TODO: can I avoid vectorize?
            # loop over non-core dims (i.e. image_id);
            # assumes function only takes arrays over core dims as input
            exclude_dims={"id"},
            # to allow dimensions that change size between input and output
        )
    )

    # Postprocess data arrays
    fused_data_arrays = _postprocess_multi_image_fused_arrays(
        position=centroid_fused_da,
        shape=shape_fused_da,
        confidence=confidence_fused_da,
        label=label_fused_da,
    )

    # Return a dataset
    return xr.Dataset(data_vars=fused_data_arrays)


def _validate_image_shape(image_shape) -> np.ndarray:
    """Validate and convert image shape to numpy array.

    Args:
        image_shape: Image dimensions as (width, height).
            Should be array-like with 2 elements.

    Returns:
        np.ndarray: Validated image shape as 1D array with 2 elements.

    Raises:
        ValueError: If image_shape cannot be converted to a valid shape.

    """
    try:
        image_shape = np.asarray(image_shape)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Cannot convert 'image_shape' to array: {e}. "
            "Expected format: (width, height) as tuple or array-like."
        ) from e

    # Flatten to handle (2,), (1,2) and (2,1) shapes
    image_shape = image_shape.flatten()
    if image_shape.shape != (2,):
        raise ValueError(
            f"'image_shape' must have exactly 2 elements (width, height), "
            f"got shape {image_shape.shape}"
        )

    return image_shape


@_check_input(ValidBboxDetectionsEnsembleDataset)
def _estimate_max_n_detections(ensemble_detections_ds: xr.Dataset) -> int:
    """Get upper bound for maximum number of boxes per image after fusion."""
    detections_w_non_nan_position = (
        ensemble_detections_ds.position.notnull().all(dim="space")
    )  # True if non-nan x and y
    return (
        detections_w_non_nan_position.sum(dim="id")
        .max(dim="image_id")
        .sum()
        .item()
    )


def _preprocess_single_image_detections(
    position: xr.DataArray,
    shape: xr.DataArray,
    confidence: xr.DataArray,
    label: xr.DataArray,
    image_width_height: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Prepare ensemble detections on a single image for fusion."""
    # Prepare boxes array --> position, shape arrays to x1y1x2y normalised
    bboxes_x1y1 = (position - shape / 2) / image_width_height[:, None, None]
    bboxes_x2y2 = (position + shape / 2) / image_width_height[:, None, None]
    bboxes_x1y1_x2y2_normalised = np.concat([bboxes_x1y1, bboxes_x2y2])
    # 4, n_annot, n_models

    # Get list of bboxes per model
    # arrays need to be tall for WBF
    n_models = bboxes_x1y1_x2y2_normalised.shape[-1]
    list_bboxes_per_model = [
        arr.squeeze()
        for arr in np.split(bboxes_x1y1_x2y2_normalised, n_models, axis=-1)
    ]
    list_confidence_per_model = [
        arr.squeeze() for arr in np.split(confidence, n_models, axis=-1)
    ]
    list_label_per_model = [
        arr.squeeze() for arr in np.split(label, n_models, axis=-1)
    ]

    # Remove rows with nan coordinates
    list_bboxes_per_model = [
        arr[:, ~np.any(np.isnan(arr), axis=0)].T
        for arr in list_bboxes_per_model
    ]
    list_confidence_per_model = [
        conf_arr[: bbox_arr.shape[0]]
        for bbox_arr, conf_arr in zip(
            list_bboxes_per_model,
            list_confidence_per_model,
            strict=True,
        )
    ]
    list_label_per_model = [
        label_arr[: bbox_arr.shape[0]]
        for bbox_arr, label_arr in zip(
            list_bboxes_per_model,
            list_label_per_model,
            strict=True,
        )
    ]

    return (
        list_bboxes_per_model,
        list_confidence_per_model,
        list_label_per_model,
    )


def _postprocess_single_image_detections(
    ensemble_x1y1_x2y2_norm,
    ensemble_scores,
    ensemble_labels,
    image_width_height,
    max_n_detections,
):
    """Postprocess fused single-image detections as dataarrays.

    Unnormalise, pad and format as data arrays.
    """
    # Undo boxes x1y1 x2y2 normalization
    ensemble_x1y1_x2y2 = ensemble_x1y1_x2y2_norm * np.tile(
        image_width_height, (1, 2)
    )

    # Combine x1y1, x2y2, scores and labels in one array
    ensemble_data = np.c_[ensemble_x1y1_x2y2, ensemble_scores, ensemble_labels]

    # Remove rows with nan coordinates
    ensemble_data = ensemble_data[
        ~np.any(np.isnan(ensemble_x1y1_x2y2), axis=1)
    ]

    # Check padding
    if ensemble_data.shape[0] > max_n_detections:
        raise ValueError(
            "Insufficient padding provided. "
            "The estimated maximum number of detections per image was set to "
            f"{max_n_detections}, "
            f"but {ensemble_data.shape[0]} detections were "
            "found in one of the images after fusion. Please increase the "
            "maximum number of detections per image."
        )

    # Pad combined array to max_n_detections
    # (this is required to concatenate across image_ids)
    ensemble_data = np.pad(
        ensemble_data,
        (
            (0, max_n_detections - ensemble_data.shape[0]),
            (0, 0),
        ),
        "constant",
        constant_values=np.nan,
    )

    # Format output as xarray dataarrays
    centroid_da, shape_da, confidence_da, label_da = (
        _single_image_detections_as_dataarrays(
            ensemble_data[:, 0:4],
            ensemble_data[:, 4],
            ensemble_data[:, 5],
        )
    )

    return centroid_da, shape_da, confidence_da, label_da


def _fuse_single_image_detections(
    fusion_function: Callable,
    position,
    shape,
    confidence: np.ndarray,
    label: np.ndarray,
    image_width_height: np.ndarray,
    max_n_detections: int,
    **fusion_kwargs: Unpack[_TypeFusionKwargs],  #  method-only kwargs
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Fuse detections across models for a single image using WBF."""
    # Prepare single image arrays for fusion
    list_bboxes_per_model, list_confidence_per_model, list_label_per_model = (
        _preprocess_single_image_detections(
            position, shape, confidence, label, image_width_height
        )
    )

    # ------------------------------------
    # Run WBF on one image
    ensemble_x1y1_x2y2_norm, ensemble_scores, ensemble_labels = (
        fusion_function(
            list_bboxes_per_model,
            list_confidence_per_model,
            list_label_per_model,
            **fusion_kwargs,
        )
    )

    # ------------------------------------

    # Format output as xarray dataarrays
    centroid_da, shape_da, confidence_da, label_da = (
        _postprocess_single_image_detections(
            ensemble_x1y1_x2y2_norm,
            ensemble_scores,
            ensemble_labels,
            image_width_height,
            max_n_detections,
        )
    )

    return centroid_da, shape_da, confidence_da, label_da


def _single_image_detections_as_dataarrays(
    x1y1_x2y2_array: np.ndarray,
    scores_array: np.ndarray,
    labels_array: np.ndarray,
    id_array: np.ndarray | None = None,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Format single image fused detections as data arrays."""
    if id_array is None:
        n_detections = x1y1_x2y2_array.shape[0]
        id_array = np.arange(n_detections)

    # Extract bbox corner coordinates
    x1y1, x2y2 = x1y1_x2y2_array[:, 0:2], x1y1_x2y2_array[:, 2:4]

    # Shared coordinates
    id_coords = {"id": id_array}
    spatial_id_coords = {"space": ["x", "y"], **id_coords}

    # Build all DataArrays
    return (
        xr.DataArray(
            (0.5 * (x1y1 + x2y2)).T,
            dims=["space", "id"],
            coords=spatial_id_coords,
        ),
        xr.DataArray(
            (x2y2 - x1y1).T, dims=["space", "id"], coords=spatial_id_coords
        ),
        xr.DataArray(scores_array, dims=["id"], coords=id_coords),
        xr.DataArray(labels_array, dims=["id"], coords=id_coords),
    )


def _postprocess_multi_image_fused_arrays(
    position: xr.DataArray,
    shape: xr.DataArray,
    confidence: xr.DataArray,
    label: xr.DataArray,
) -> dict:
    """Postprocess fused data arrays on multiple images after fusion."""
    data_arrays = [position, shape, confidence, label]

    # Remove extra padding across annotations
    position_da, shape_da, confidence_da, label_da = [
        da.dropna(dim="id", how="all") for da in data_arrays
    ]

    # Pad labels with -1 rather than nan
    label_da = label_da.fillna(-1).astype(int)

    # Assign id coordinates to data arrays
    # (these are lost after apply_ufunc because exclude_dims is used)
    n_max_detections = position_da.sizes["id"]
    id_coords = np.arange(n_max_detections)
    return {
        "position": position_da.assign_coords(id=id_coords),
        "shape": shape_da.assign_coords(id=id_coords),
        "confidence": confidence_da.assign_coords(id=id_coords),
        "label": label_da.assign_coords(id=id_coords),
    }
