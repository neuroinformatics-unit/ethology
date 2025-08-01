"""Utils for ensembles of detectors."""

import numpy as np
import xarray as xr
from ensemble_boxes import weighted_boxes_fusion

from ethology.detectors.utils import (
    detections_x1y1_x2y2_as_da_tuple,
)

# def soft_nms_wrapper_arrays(
#     bboxes_x1y1: np.ndarray,
#     bboxes_x2y2: np.ndarray,  # model, annot, 4
#     confidence: np.ndarray,  # model, annot
#     label: np.ndarray,  # model, annot
#     image_width_height: np.ndarray,  # = np.array([4096, 2160]),
#     iou_thr_ensemble: float = 0.5,
#     skip_box_thr: float = 0.0001,
#     max_n_detections: int = 300,
# ) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
#     """Wrap weighted boxes fusion to receive arrays as input.

#     Parameters
#     ----------
#     bboxes_x1y1: np.ndarray
#         Detected bounding boxes in a single imagein x1y1 format, with shape
#         n_models, n_annotations, 2.
#     bboxes_x2y2: np.ndarray
#         Detected bounding boxes in a single image in x2y2 format, with shape
#         n_models, n_annotations, 2.
#     confidence: np.ndarray
#         Confidence scores for each bounding box, with shape
#         n_models, n_annotations.
#     label: np.ndarray
#         Labels for each bounding box, with shape n_models, n_annotations.
#     image_width_height: np.ndarray
#         Width and height of the image, with shape 2.
#     iou_thr_ensemble: float
#         IoU threshold for detections to be considered for fusion.
#     skip_box_thr: float
#         Threshold for skipping boxes with confidence below this value.
#     max_n_detections: int
#         Fused bounding boxes arrays are padded to this total number of boxes.
#         Its value should be larger than the expected maximum number of
#         detections per image after fusing across models.

#     Returns
#     -------
#     tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]
#         Tuple of xr.DataArrays containing the fused detections. The arrays
#         are padded to max_n_detections and contain the data for the centroid,
#         shape, confidence and label of the fused detections.

#     """
#     # Prepare bboxes for WBF
#     bboxes_x1y1_x2y2_normalised = np.concat(
#         [bboxes_x1y1, bboxes_x2y2], axis=-1
#     ) / np.tile(image_width_height, (1, 2))  # [:, :, :, None]

#     # ------------------------------------
#     # Run WBF
#     ensemble_x1y1_x2y2_norm, ensemble_scores, ensemble_labels = soft_nms(
#         bboxes_x1y1_x2y2_normalised,
#         confidence,
#         label,
#         iou_thr=iou_thr_ensemble,
#         thresh=skip_box_thr,  # threshold for boxes to keep
#         method=2,  # 1-linear soft-NMS, 2-gaussian soft-NMS, 3-standard NMS
#         sigma=0.5,  # sigma for gaussian soft-NMS
#     )

#     # ------------------------------------
#     # Undo x1y1 x2y2 normalization
#     ensemble_x1y1_x2y2 = ensemble_x1y1_x2y2_norm * np.tile(
#         image_width_height, (1, 2)
#     )

#     # Combine x1y1, x2y2, scores and labels in one array
#     ensemble_x1y2_x2y2_scores_labels = np.c_[
#         ensemble_x1y1_x2y2, ensemble_scores, ensemble_labels
#     ]

#     # Remove rows with nan coordinates
#     slc_nan_rows = np.any(np.isnan(ensemble_x1y1_x2y2), axis=1)
#     ensemble_x1y2_x2y2_scores_labels = ensemble_x1y2_x2y2_scores_labels[
#         ~slc_nan_rows
#     ]

#     # Pad combined array to max_n_detections
#     # (this is required to concatenate across image_ids
#     ensemble_x1y2_x2y2_scores_labels = np.pad(
#         ensemble_x1y2_x2y2_scores_labels,
#         (
#             (0,
# max_n_detections - ensemble_x1y2_x2y2_scores_labels.shape[0]),
#             (0, 0),
#         ),
#         "constant",
#         constant_values=np.nan,
#     )

#     # Format output as xarray dataarrays
#     centroid, shape, confidence, label = detections_x1y1_x2y2_as_da_tuple(
#         ensemble_x1y2_x2y2_scores_labels[:, 0:4],
#         ensemble_x1y2_x2y2_scores_labels[:, 4],
#         ensemble_x1y2_x2y2_scores_labels[:, 5],
#     )

#     return centroid, shape, confidence, label


def wbf_wrapper_arrays(
    bboxes_x1y1: np.ndarray,
    bboxes_x2y2: np.ndarray,  # model, annot, 4
    confidence: np.ndarray,  # model, annot
    label: np.ndarray,  # model, annot
    image_width_height: np.ndarray,  # = np.array([4096, 2160]),
    iou_thr_ensemble: float = 0.5,
    skip_box_thr: float = 0.0001,
    max_n_detections: int = 300,
    confidence_th_post_fusion: float = 0.7,
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]:
    """Wrap weighted boxes fusion to receive arrays as input.

    Parameters
    ----------
    bboxes_x1y1: np.ndarray
        Detected bounding boxes in a single imagein x1y1 format, with shape
        n_models, n_annotations, 2.
    bboxes_x2y2: np.ndarray
        Detected bounding boxes in a single image in x2y2 format, with shape
        n_models, n_annotations, 2.
    confidence: np.ndarray
        Confidence scores for each bounding box, with shape
        n_models, n_annotations.
    label: np.ndarray
        Labels for each bounding box, with shape n_models, n_annotations.
    image_width_height: np.ndarray
        Width and height of the image, with shape 2.
    iou_thr_ensemble: float
        IoU threshold for detections to be considered for fusion.
    skip_box_thr: float
        Threshold for skipping boxes with confidence below this value.
    max_n_detections: int
        Fused bounding boxes arrays are padded to this total number of boxes.
        Its value should be larger than the expected maximum number of
        detections per image after fusing across models.
    confidence_th_post_fusion: float
        Threshold for removing fused detections whose confidence is below
        this value.

    Returns
    -------
    tuple[xr.DataArray, xr.DataArray, xr.DataArray, xr.DataArray]
        Tuple of xr.DataArrays containing the fused detections. The arrays
        are padded to max_n_detections and contain the data for the centroid,
        shape, confidence and label of the fused detections.

    """
    # Prepare bboxes for WBF
    bboxes_x1y1_x2y2_normalised = np.concat(
        [bboxes_x1y1, bboxes_x2y2], axis=-1
    ) / np.tile(image_width_height, (1, 2))  # [:, :, :, None]

    # Remove rows with nan coordinates
    n_models = bboxes_x1y1_x2y2_normalised.shape[0]
    list_bboxes_per_model = [
        arr.squeeze()
        for arr in np.split(bboxes_x1y1_x2y2_normalised, n_models, axis=0)
    ]
    list_bboxes_per_model = [
        arr[~np.any(np.isnan(arr), axis=1), :] for arr in list_bboxes_per_model
    ]
    list_confidence_per_model = [
        conf_arr.squeeze()[: bbox_arr.shape[0]]
        for bbox_arr, conf_arr in zip(
            list_bboxes_per_model,
            np.split(confidence, n_models, axis=0),
            strict=True,
        )
    ]
    list_label_per_model = [
        label_arr.squeeze()[: bbox_arr.shape[0]]
        for bbox_arr, label_arr in zip(
            list_bboxes_per_model,
            np.split(label, n_models, axis=0),
            strict=True,
        )
    ]
    # ------------------------------------
    # Run WBF
    ensemble_x1y1_x2y2_norm, ensemble_scores, ensemble_labels = (
        weighted_boxes_fusion(
            list_bboxes_per_model,
            list_confidence_per_model,
            list_label_per_model,
            iou_thr=iou_thr_ensemble,
            skip_box_thr=skip_box_thr,
        )
    )

    # ------------------------------------
    # Undo x1y1 x2y2 normalization
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

    # Remove rows with confidence below threshold
    ensemble_x1y2_x2y2_scores_labels = ensemble_x1y2_x2y2_scores_labels[
        ensemble_x1y2_x2y2_scores_labels[:, 4] > confidence_th_post_fusion
    ]

    # Pad combined array to max_n_detections
    # (this is required to concatenate across image_ids
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
    centroid, shape, confidence, label = detections_x1y1_x2y2_as_da_tuple(
        ensemble_x1y2_x2y2_scores_labels[:, 0:4],
        ensemble_x1y2_x2y2_scores_labels[:, 4],
        ensemble_x1y2_x2y2_scores_labels[:, 5],
    )

    return centroid, shape, confidence, label


def combine_detections_across_models_wbf(
    all_models_detections_ds: xr.Dataset,
    kwargs_wbf: dict,
) -> xr.Dataset:
    """Combine detections across models using weighted boxes fusion.

    Parameters
    ----------
    all_models_detections_ds: xr.Dataset
        Dataset containing the detections from all models. It should contain
        the following variables: xy_min, xy_max, confidence, label.
    kwargs_wbf: dict
        Keyword arguments for the weighted boxes fusion approach. It should
        contain the following keys:
        - iou_thr_ensemble: IoU threshold for detections to be considered
        for fusion.
        - skip_box_thr: Threshold for skipping boxes with confidence below
        this value.
        - max_n_detections: Fused bounding boxes arrays are padded to this
        total number of boxes. Its value should be larger than the expected
        maximum number of detections per image after fusing across models.
        - confidence_th_post_fusion: Threshold for removing fused detections
        whose confidence is below this value.

    Returns
    -------
    xr.Dataset
        Detections dataset containing the fused detections.

    """
    # Prepare kwargs
    kwargs_wbf["image_width_height"] = np.array(
        [
            all_models_detections_ds.attrs[img_size]
            for img_size in ["image_width", "image_height"]
        ]
    )

    # Run WBF vectorized
    centroid_fused, shape_fused, confidence_fused, label_fused = (
        xr.apply_ufunc(
            wbf_wrapper_arrays,  # ------------#wbf_wrapper_arrays,
            all_models_detections_ds.xy_min,  # .data array is passed
            all_models_detections_ds.xy_max,
            all_models_detections_ds.confidence,
            all_models_detections_ds.label,
            kwargs=kwargs_wbf,
            input_core_dims=[  # do not broadcast across these
                ["model", "id", "space"],
                ["model", "id", "space"],
                ["model", "id"],
                ["model", "id"],
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

    # Remove pad across annotations
    centroid_fused = centroid_fused.dropna(dim="id", how="all")
    shape_fused = shape_fused.dropna(dim="id", how="all")
    confidence_fused = confidence_fused.dropna(dim="id", how="all")
    label_fused = label_fused.dropna(dim="id", how="all")

    # Pad labels with -1 rather than nan
    label_fused = label_fused.fillna(-1).astype(int)

    # Return a dataset
    # FIX: why is id not a coordinate in the output dataset?
    # FIX: order of dimensions should be image_id, space, id
    return xr.Dataset(
        data_vars={
            "position": centroid_fused,
            "shape": shape_fused,
            "confidence": confidence_fused,
            "label": label_fused,
        }
    )


# def apply_nms_to_detections_ds(
#     detections_ds: xr.Dataset,
#     nms_iou_threshold: float = 0.5,
# ) -> xr.Dataset:
#     """Apply non-maximum suppression to detections dataset."""

#     def padded_batched_nms(
#         bboxes: torch.Tensor,
#         scores: torch.Tensor,
#         labels: torch.Tensor,
#         iou_threshold: float,
#     ) -> torch.Tensor:
#         n_input_detections = bboxes.shape[0]
#         idcs_to_keep = torchvision.ops.batched_nms(
#             bboxes, scores, labels, iou_threshold
#         )
#         # pad with -1
#         idcs_to_keep = torch.nn.functional.pad(
#             idcs_to_keep,
#             (0, n_input_detections - idcs_to_keep.shape[0]),
#             value=-1,
#         )
#         return idcs_to_keep

#     # Add xy_min and xy_max if not present
#     if all(
#         [
#             var_str not in detections_ds.variables
#             for var_str in ["xy_min", "xy_max"]
#         ]
#     ):
#         detections_ds = add_bboxes_min_max_corners(detections_ds)

#     # Prepare input for nms
#     ensemble_x1y1_x2y2 = xr.concat(
#         [detections_ds.xy_min, detections_ds.xy_max], dim="space"
#     ).transpose("image_id", "id", "space")

#     # Apply nms
#     nms_vectorized = torch.vmap(
#         padded_batched_nms, in_dims=(0, 0, 0, None)
#     )
#     idcs_to_keep = nms_vectorized(
#         torch.from_numpy(ensemble_x1y1_x2y2.data),
#         torch.from_numpy(detections_ds.confidence.data),
#         torch.from_numpy(detections_ds.label.data),
#         nms_iou_threshold,
#     )  # idcs per image, sorted by confidence

#     # Return detections dataset with only the detections that are kept
#     return detections_ds.sel(id=idcs_to_keep)
