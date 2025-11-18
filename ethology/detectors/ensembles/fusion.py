"""Wrappers around ensemble-boxes fusion functions."""

import numpy as np
import xarray as xr
from ensemble_boxes import weighted_boxes_fusion, nms


# ----------- Helper functions ---------------------------
def _preprocess_single_image_detections(
    position: xr.DataArray,
    shape: xr.DataArray,
    confidence: xr.DataArray,
    label: xr.DataArray,
    image_width_height: np.ndarray,
) -> list[np.ndarray]:
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
    """Unnormalise, pad and format fused single-image detections as data arrays."""
    # Undo boxes x1y1 x2y2 normalization
    ensemble_x1y1_x2y2 = ensemble_x1y1_x2y2_norm * np.tile(
        image_width_height, (1, 2)
    )

    # Combine x1y1, x2y2, scores and labels in one array
    ensemble_x1y2_x2y2_scores_labels = np.c_[
        ensemble_x1y1_x2y2, ensemble_scores, ensemble_labels
    ]

    # Remove rows with nan coordinates
    ensemble_x1y2_x2y2_scores_labels = ensemble_x1y2_x2y2_scores_labels[
        ~np.any(np.isnan(ensemble_x1y1_x2y2), axis=1)
    ]

    # Pad combined array to max_n_detections
    # (this is required to concatenate across image_ids)
    ensemble_x1y2_x2y2_scores_labels = np.pad(
        ensemble_x1y2_x2y2_scores_labels,
        (
            (0, max_n_detections - ensemble_x1y2_x2y2_scores_labels.shape[0]),
            (0, 0),
        ),
        "constant",
        constant_values=np.nan,
    )

    # Format output as xarray dataarrays
    centroid_da, shape_da, confidence_da, label_da = (
        _single_image_detections_as_dataarrays(
            ensemble_x1y2_x2y2_scores_labels[:, 0:4],
            ensemble_x1y2_x2y2_scores_labels[:, 4],
            ensemble_x1y2_x2y2_scores_labels[:, 5],
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

    # Remove padding across annotations
    position_da, shape_da, confidence_da, label_da = [
        da.dropna(dim="id", how="all") for da in data_arrays
    ]

    # Pad labels with -1 rather than nan
    label_da = label_da.fillna(-1).astype(int)

    return {
        "position": position_da,
        "shape": shape_da,
        "confidence": confidence_da,
        "label": label_da,
    }


# -------------------------------------


def fuse_ensemble_detections_WBF(
    ensemble_detections_ds: xr.Dataset,
    image_width_height: np.ndarray,
    max_n_detections: int,
    wbf_kwargs: dict,
    # iou_thr_ensemble: float = 0.5,
    # skip_box_thr: float = 0.0001,
    # max_n_detections: int = 300,
) -> xr.Dataset:
    """Fuse ensemble detections across models using WBF."""

    # Run WBF across image_id
    centroid_fused_da, shape_fused_da, confidence_fused_da, label_fused_da = (
        xr.apply_ufunc(
            _fuse_single_image_detections_WBF,
            ensemble_detections_ds.position,  # .data array is passed
            ensemble_detections_ds.shape,
            ensemble_detections_ds.confidence,
            ensemble_detections_ds.label,
            kwargs={
                "image_width_height": image_width_height,
                "max_n_detections": max_n_detections,
                **wbf_kwargs,
            },
            input_core_dims=[  # do not broadcast across these
                ["space", "id", "model"],
                ["space", "id", "model"],
                ["id", "model"],
                ["id", "model"],
            ],
            output_core_dims=[
                ["space", "id"],
                ["space", "id"],
                ["id"],
                ["id"],
            ],
            vectorize=True,
            # loop over non-core dims (i.e. image_id);
            # assumes function only takes arrays over core dims as input
            exclude_dims={"id"},
            # to allow dimensions that change size btw input and output
        )
    )

    # Post process data arrays
    fused_data_arrays = {
        "position": centroid_fused_da,
        "shape": shape_fused_da,
        "confidence": confidence_fused_da,
        "label": label_fused_da,
    }
    fused_data_arrays = _postprocess_multi_image_fused_arrays(
        **fused_data_arrays
    )

    # Return a dataset
    # FIX: why is id not a coordinate in the output dataset?
    # FIX: order of dimensions should be image_id, space, id
    return xr.Dataset(data_vars=fused_data_arrays)


def fuse_ensemble_detections_NMS(
    ensemble_detections_ds: xr.Dataset,
    image_width_height: np.ndarray,
    max_n_detections: int,
    nms_kwargs: dict,
    # iou_thr_ensemble: float = 0.5,
    # skip_box_thr: float = 0.0001,
    # max_n_detections: int = 300,
) -> xr.Dataset:
    """Fuse ensemble detections across models using WBF."""

        # Run WBF across image_id
    centroid_fused_da, shape_fused_da, confidence_fused_da, label_fused_da = (
        xr.apply_ufunc(
            _fuse_single_image_detections_NMS,
            ensemble_detections_ds.position,  # .data array is passed
            ensemble_detections_ds.shape,
            ensemble_detections_ds.confidence,
            ensemble_detections_ds.label,
            kwargs={
                "image_width_height": image_width_height,
                "max_n_detections": max_n_detections,
                **nms_kwargs,
            },
            input_core_dims=[  # do not broadcast across these
                ["space", "id", "model"],
                ["space", "id", "model"],
                ["id", "model"],
                ["id", "model"],
            ],
            output_core_dims=[
                ["space", "id"],
                ["space", "id"],
                ["id"],
                ["id"],
            ],
            vectorize=True,
            # loop over non-core dims (i.e. image_id);
            # assumes function only takes arrays over core dims as input
            exclude_dims={"id"},
            # to allow dimensions that change size btw input and output
        )
    )

    # Post process data arrays
    fused_data_arrays = {
        "position": centroid_fused_da,
        "shape": shape_fused_da,
        "confidence": confidence_fused_da,
        "label": label_fused_da,
    }
    fused_data_arrays = _postprocess_multi_image_fused_arrays(
        **fused_data_arrays
    )

    # Return a dataset
    # FIX: why is id not a coordinate in the output dataset?
    # FIX: order of dimensions should be image_id, space, id
    return xr.Dataset(data_vars=fused_data_arrays)


# --------------- Single image ---------------------------
def _fuse_single_image_detections_WBF(
    position,  # bboxes_x1y1: np.ndarray,  # model, annot, 4
    shape,  # bboxes_x2y2: np.ndarray,  # model, annot, 4
    confidence: np.ndarray,  # model, annot
    label: np.ndarray,  # model, annot
    image_width_height: np.ndarray,  # = np.array([4096, 2160]),
    max_n_detections: int,
    **wbf_kwargs: dict,  # WBF only kwargs
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
        weighted_boxes_fusion(
            list_bboxes_per_model,
            list_confidence_per_model,
            list_label_per_model,
            **wbf_kwargs,
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


def _fuse_single_image_detections_NMS(
    position,  # bboxes_x1y1: np.ndarray,  # model, annot, 4
    shape,  # bboxes_x2y2: np.ndarray,  # model, annot, 4
    confidence: np.ndarray,  # model, annot
    label: np.ndarray,  # model, annot
    image_width_height: np.ndarray,  # = np.array([4096, 2160]),
    max_n_detections: int,
    **nms_kwargs: dict,  # NMS only kwargs
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Fuse detections across models for a single image using NMS."""
    # Prepare single image arrays for fusion
    list_bboxes_per_model, list_confidence_per_model, list_label_per_model = (
        _preprocess_single_image_detections(
            position, shape, confidence, label, image_width_height
        )
    )

    # ------------------------------------
    # Run WBF on one image
    ensemble_x1y1_x2y2_norm, ensemble_scores, ensemble_labels = nms(
        list_bboxes_per_model,
        list_confidence_per_model,
        list_label_per_model,
        **nms_kwargs,
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
