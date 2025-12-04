"""Wrappers around ensemble-boxes fusion functions."""

from collections.abc import Callable
from functools import partial
from typing import Literal, TypeAlias, TypedDict, Unpack

import ensemble_boxes
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from tqdm import tqdm

from ethology.detectors.ensembles.utils import (
    _centroid_shape_to_corners,
    _corners_to_centroid_shape,
)
from ethology.validators.detections import (
    ValidBboxDetectionsDataset,
    ValidBboxDetectionsEnsembleDataset,
)
from ethology.validators.utils import _check_input, _check_output

# ------------------- Supported fusion methods ------------------
# from ensemble_boxes
VALID_FUSION_METHODS = {
    "weighted_boxes_fusion": ensemble_boxes.weighted_boxes_fusion,
    "nms": ensemble_boxes.nms,
    "soft_nms": ensemble_boxes.soft_nms,
    "non_maxium_weighted": ensemble_boxes.non_maximum_weighted,
}


#  ------------------ Custom types  ----------------------
TypeFusionMethod = Literal[
    "weighted_boxes_fusion",
    "nms",
    "soft_nms",
    "non_maxium_weighted",
]

TupleFourDataArrays: TypeAlias = tuple[
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
    xr.DataArray,
]


class _TypeFusionMethodKwargs(TypedDict, total=False):
    """Type hints for fusion method keyword arguments.

    Parameters for methods as described in the ensemble_boxes documentation.
    See https://github.com/ZFTurbo/Weighted-Boxes-Fusion
    """

    weights: list[float] | None
    iou_thr: float
    skip_box_thr: float
    sigma: float
    thresh: float
    conf_type: Literal["avg", "box_and_model_avg", "absent_model_aware_avg"]
    allows_overflow: bool


# ----------------------------------


@_check_input(ValidBboxDetectionsEnsembleDataset)
@_check_output(ValidBboxDetectionsDataset)
def fuse_detections(
    ensemble_detections_ds: xr.Dataset,
    fusion_method: TypeFusionMethod,
    fusion_method_kwargs: dict | None = None,
    max_n_detections: int | None = None,
    n_workers: int | None = -1,
) -> xr.Dataset:
    """Fuse ensemble detections across models using the selected method.

    You can set a max_n_detections if upper bound is known a prior to
    reduce memory usage. n_workers: number of workers for joblib.Parallel

    """
    # Check if image_width_height defined in dataset
    image_shape = ensemble_detections_ds.attrs.get("image_shape")
    if image_shape is None:
        raise KeyError(
            "Required attribute 'image_shape' not found in the dataset "
            "attributes. Please ensure the dataset has 'image_shape' "
            "(width, height in pixels) in its attributes."
        )
    image_width_height = _validate_image_shape(image_shape)

    # Compute upper bound of max_n_detections
    if not max_n_detections:
        max_n_detections = _estimate_max_n_detections(ensemble_detections_ds)

    # Build single-image partial function for the selected fusion method
    if fusion_method not in VALID_FUSION_METHODS:
        raise ValueError(
            f"Invalid fusion method: {fusion_method}. "
            f"Valid methods are: {list(VALID_FUSION_METHODS.keys())}"
        )
    fusion_function = VALID_FUSION_METHODS[fusion_method]
    _fuse_single_image_detections_partial = partial(
        _fuse_single_image_detections, fusion_function
    )

    # Parallelise fusion across image_id
    results_per_img_id = Parallel(n_jobs=n_workers)(
        delayed(_fuse_single_image_detections_partial)(
            ensemble_detections_ds.position.sel(image_id=img_id).values,
            ensemble_detections_ds.shape.sel(image_id=img_id).values,
            ensemble_detections_ds.confidence.sel(image_id=img_id).values,
            ensemble_detections_ds.label.sel(image_id=img_id).values,
            image_width_height,
            max_n_detections,
            **fusion_method_kwargs,
        )
        for img_id in tqdm(ensemble_detections_ds.image_id)
    )

    # Postprocess data arrays
    fused_detections_ds = _postprocess_multi_image_fused_arrays(
        results_per_img_id, ensemble_detections_ds.image_id
    )

    return fused_detections_ds


# ------- Multi image fusion ------------------


@_check_output(ValidBboxDetectionsDataset)
def _postprocess_multi_image_fused_arrays(
    results_per_img_id: list[TupleFourDataArrays],
    list_img_id: list,
) -> xr.Dataset:
    """Postprocess fused data arrays on multiple images after fusion.

    Fix padding and assign id coordinates.
    """
    # Transpose results from list-of-tuples to tuple-of-lists
    da_names = ("position", "shape", "confidence", "label")
    da_lists = zip(*results_per_img_id, strict=True)

    # Concatenate lists of dataarrays along image_id dimension and
    # remove extra padding in "id" dimension
    fused_da_dict = {}
    for da_str, list_da in zip(da_names, da_lists, strict=True):
        fused_da_dict[da_str] = xr.concat(
            list_da, pd.Index(list_img_id, name="image_id")
        ).dropna(dim="id", how="all")

    # Pad labels with -1 rather than nan
    fused_da_dict["label"] = fused_da_dict["label"].fillna(-1).astype(int)

    return xr.Dataset(data_vars=fused_da_dict)


def _validate_image_shape(image_shape) -> np.ndarray:
    """Validate and cast image shape as numpy array."""
    # Try casting as numpy array
    try:
        image_shape = np.asarray(image_shape)
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"Cannot convert 'image_shape' to array: {e}. "
            "Expected format: (width, height) as tuple or array-like."
        ) from e

    # Check number of elements in array
    if image_shape.size != 2:
        raise ValueError(
            f"'image_shape' must have exactly 2 elements (width, height), "
            f"got shape {image_shape.shape}"
        )
    return image_shape


@_check_input(ValidBboxDetectionsEnsembleDataset)
def _estimate_max_n_detections(ensemble_detections_ds: xr.Dataset) -> int:
    """Get upper bound for maximum number of boxes per image after fusion.

    We assume no detections are fused and all images have as many
    detections as the maximum number of non-nan detections per image.
    """
    detections_w_non_nan_position = (
        ensemble_detections_ds.position.notnull().all(dim="space")
    )  # True if non-nan x and y
    return (
        detections_w_non_nan_position.sum(dim="id")
        .max(dim="image_id")
        .sum()
        .item()
    )


# ------- Single image fusion ------------------


def _fuse_single_image_detections(
    fusion_function: Callable,
    position: np.ndarray,
    shape: np.ndarray,
    confidence: np.ndarray,
    label: np.ndarray,
    image_width_height: np.ndarray,
    max_n_detections: int,
    **fusion_kwargs: Unpack[_TypeFusionMethodKwargs],  #  method-only kwargs
) -> TupleFourDataArrays:
    """Fuse detections for a single image with selected method."""
    # Prepare single image arrays for fusion
    list_bboxes_per_model, list_confidence_per_model, list_label_per_model = (
        _preprocess_single_image_detections(
            position, shape, confidence, label, image_width_height
        )
    )

    # Run fusion method on one image
    ensemble_x1y1_x2y2_norm, ensemble_scores, ensemble_labels = (
        fusion_function(
            list_bboxes_per_model,
            list_confidence_per_model,
            list_label_per_model,
            **fusion_kwargs,
        )
    )

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


def _preprocess_single_image_detections(
    position: xr.DataArray,
    shape: xr.DataArray,
    confidence: xr.DataArray,
    label: xr.DataArray,
    image_width_height: np.ndarray,
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Prepare detections of an ensemble on a single image for fusion."""
    # Prepare boxes array
    # transform position and shape arrays to x1y1x2y normalised
    x1y1, x2y2 = _centroid_shape_to_corners(position, shape)
    bboxes_x1y1 = x1y1 / image_width_height[:, None, None]
    bboxes_x2y2 = x2y2 / image_width_height[:, None, None]
    bboxes_x1y1_x2y2_normalised = np.transpose(
        np.concat(
            [bboxes_x1y1, bboxes_x2y2]
        ),  # shape: 4, max_n_annotations_per_frame, n_models
        (1, 0, 2),  # shape: max_n_annotations_per_frame, 4, n_models
    )

    # --------------------
    # Get list of bboxes per model
    # arrays need to be tall for fusion methods
    n_models = bboxes_x1y1_x2y2_normalised.shape[-1]
    list_x1y1_x2y2_norm_per_model = [
        arr.squeeze()
        for arr in np.split(bboxes_x1y1_x2y2_normalised, n_models, axis=-1)
    ]
    list_confidence_per_model = [
        arr.squeeze() for arr in np.split(confidence, n_models, axis=-1)
    ]
    list_label_per_model = [
        arr.squeeze() for arr in np.split(label, n_models, axis=-1)
    ]
    # --------------------

    # Remove rows with nan coordinates and return lists of arrays
    list_non_nan_bboxes_per_model = [
        sum(~np.any(np.isnan(arr), axis=1))
        for arr in list_x1y1_x2y2_norm_per_model
    ]
    return (
        _chop_end_of_array(
            list_x1y1_x2y2_norm_per_model, list_non_nan_bboxes_per_model
        ),
        _chop_end_of_array(
            list_confidence_per_model, list_non_nan_bboxes_per_model
        ),
        _chop_end_of_array(
            list_label_per_model, list_non_nan_bboxes_per_model
        ),
    )


def _chop_end_of_array(
    list_arrays: list[np.ndarray], list_end_lengths: list[int]
) -> list[np.ndarray]:
    """Chop end of arrays in list to desired length along first dimension."""
    return [
        arr[:n] for arr, n in zip(list_arrays, list_end_lengths, strict=True)
    ]


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

    # Get 1d array for non-nan boxes
    bool_non_nan_array = ~np.any(np.isnan(ensemble_x1y1_x2y2), axis=1)
    n_non_nan_boxes = bool_non_nan_array.sum()
    if n_non_nan_boxes > max_n_detections:
        raise ValueError(
            "Insufficient padding provided. "
            "The estimated maximum number of detections per image was set to "
            f"{max_n_detections}, "
            f"but {n_non_nan_boxes} detections were "
            "found in one of the images after fusion. Please increase the "
            "maximum number of detections per image."
        )

    # Retain non-nan boxes only and pad each array
    return _parse_single_image_detections_as_dataarrays(
        *(
            _remove_nan_and_pad_to_max(
                arr, bool_non_nan_array, max_n_detections
            )
            for arr in (ensemble_x1y1_x2y2, ensemble_scores, ensemble_labels)
        ),
    )


def _remove_nan_and_pad_to_max(
    input_array, mask_non_nan_rows, max_n_detections, fill_value=np.nan
):
    """Remove non-nan from input array and pad, all along first dimension."""
    # Initialise array with nans
    padded_array = np.full(
        (max_n_detections, *input_array.shape[1:]),
        fill_value,
        dtype=input_array.dtype,
    )
    # Replace top "mask_non_nan_rows.sum()" chunk with non-nan values from
    # input array
    padded_array[: mask_non_nan_rows.sum()] = input_array[mask_non_nan_rows]
    return padded_array


def _parse_single_image_detections_as_dataarrays(
    x1y1_x2y2_array: np.ndarray,
    scores_array: np.ndarray,
    labels_array: np.ndarray,
    id_array: np.ndarray | None = None,
) -> TupleFourDataArrays:
    """Format array of single image fused results as data arrays."""
    if id_array is None:
        n_detections = x1y1_x2y2_array.shape[0]
        id_array = np.arange(n_detections)

    # Extract bbox centre and shape
    centroid, shape = _corners_to_centroid_shape(
        x1y1_x2y2_array[:, 0:2], x1y1_x2y2_array[:, 2:4]
    )

    # Shared coordinates
    id_coords = {"id": id_array}
    spatial_id_coords = {"space": ["x", "y"], **id_coords}

    # Build all DataArrays
    return (
        xr.DataArray(
            centroid.T,
            dims=["space", "id"],
            coords=spatial_id_coords,
        ),
        xr.DataArray(shape.T, dims=["space", "id"], coords=spatial_id_coords),
        xr.DataArray(scores_array, dims=["id"], coords=id_coords),
        xr.DataArray(labels_array, dims=["id"], coords=id_coords),
    )
