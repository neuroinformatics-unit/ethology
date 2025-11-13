"""Utilities for evaluating detectors."""

import numpy as np
import torch
import torchvision.ops as ops
import xarray as xr
from scipy.optimize import linear_sum_assignment


def evaluate_detections_hungarian_ds(
    pred_bboxes_ds: xr.Dataset,
    gt_bboxes_ds: xr.Dataset,
    iou_threshold: float,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Compute true positives, false positives, and missed detections.

    Uses Hungarian algorithm for matching.
    """
    # Add xy_min and xy_max if not present
    if all(
        [
            var_str not in pred_bboxes_ds.variables
            for var_str in ["xy_min", "xy_max"]
        ]
    ):
        pred_bboxes_ds = _add_bboxes_min_max_corners(pred_bboxes_ds)

    if all(
        [
            var_str not in gt_bboxes_ds.variables
            for var_str in ["xy_min", "xy_max"]
        ]
    ):
        gt_bboxes_ds = _add_bboxes_min_max_corners(gt_bboxes_ds)

    # Prepare input for hungarian
    pred_bboxes_x1y1_x2y2 = xr.concat(
        [pred_bboxes_ds.xy_min, pred_bboxes_ds.xy_max], dim="space"
    ).transpose("image_id", "id", "space")

    gt_bboxes_x1y1_x2y2 = xr.concat(
        [gt_bboxes_ds.xy_min, gt_bboxes_ds.xy_max], dim="space"
    ).transpose("image_id", "id", "space")

    # rename id dimension in gt_bboxes_x1y1_x2y2
    gt_bboxes_x1y1_x2y2 = gt_bboxes_x1y1_x2y2.rename({"id": "id_gt"})

    # Run hungarian vectorized
    tp_array, fp_array, md_array, iou_tp_array = xr.apply_ufunc(
        _evaluate_detections_hungarian_arrays,
        pred_bboxes_x1y1_x2y2,
        gt_bboxes_x1y1_x2y2,
        kwargs={"iou_threshold": iou_threshold},
        input_core_dims=[
            ["id", "space"],
            ["id_gt", "space"],
        ],
        output_core_dims=[
            ["id"],
            ["id"],
            ["id_gt"],
            ["id"],
        ],
        vectorize=True,
        exclude_dims={"id", "id_gt"},
    )

    # Add to datasets
    pred_bboxes_ds["tp"] = xr.DataArray(tp_array, dims=["image_id", "id"])
    pred_bboxes_ds["fp"] = xr.DataArray(fp_array, dims=["image_id", "id"])
    pred_bboxes_ds["iou_tp"] = xr.DataArray(
        iou_tp_array, dims=["image_id", "id"]
    )

    # rename id dimension in md_array
    md_array = md_array.rename({"id_gt": "id"})
    gt_bboxes_ds["md"] = xr.DataArray(md_array, dims=["image_id", "id"])

    return pred_bboxes_ds, gt_bboxes_ds


def _evaluate_detections_hungarian_arrays(
    pred_bboxes: np.ndarray, gt_bboxes: np.ndarray, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute true positives, false positives, and missed detections.

    Uses Hungarian algorithm for matching and takes arrays of bboxes as input
    in x1y1x2y2 format.

    Parameters
    ----------
    pred_bboxes : np.ndarray
        An array of prediction bounding boxes with the first four columns being
        the coordinates of the bounding box in the format [x1, y1, x2, y2]
    gt_bboxes : np.ndarray
        An array of ground truth bounding boxes with the first four columns
        being the coordinates of the bounding box in the format
        [x1, y1, x2, y2]
    iou_threshold : float
        IoU threshold for considering a detection as true positive

    Returns
    -------
    tuple
        A tuple of four boolean arrays:
        - true_positives: True for each predicted bbox that is a true positive
        - false_positives: True for each predicted bbox that is a false
        positive
        - missed_detections: True for each ground truth bbox that is missed
        - true_positives_iou: IoU of each true positive

    Notes
    -----
    The output arrays are padded with False to the length of the original
    arrays. This means that for example where the true_positives array is
    False, that does not necessarily mean that the prediction is a false
    positive. The same applies for the true_positives_iou array, which is
    padded with nan.

    """
    # Remove nan values
    n_pred_bboxes_padded = pred_bboxes.shape[0]
    n_gt_bboxes_padded = gt_bboxes.shape[0]
    pred_bboxes = pred_bboxes[~np.isnan(pred_bboxes).any(axis=1), :]
    gt_bboxes = gt_bboxes[~np.isnan(gt_bboxes).any(axis=1), :]

    # Initialize output arrays
    true_positives = np.zeros(len(pred_bboxes), dtype=bool)
    false_positives = np.zeros(len(pred_bboxes), dtype=bool)
    matched_gts = np.zeros(len(gt_bboxes), dtype=bool)
    missed_detections = np.zeros(len(gt_bboxes), dtype=bool)  # unmatched gts

    true_positives_iou = np.zeros(len(pred_bboxes), dtype=float)

    # cast as a tensor if not already
    if not isinstance(pred_bboxes, torch.Tensor):
        pred_bboxes = torch.from_numpy(pred_bboxes).float()
    if not isinstance(gt_bboxes, torch.Tensor):
        gt_bboxes = torch.from_numpy(gt_bboxes).float()

    if len(pred_bboxes) > 0 and len(gt_bboxes) > 0:
        # Compute IoU matrix (pred_bboxes x gt_bboxes)
        iou_matrix = ops.box_iou(pred_bboxes[:, :4], gt_bboxes).cpu().numpy()
        # iou_matrix[np.isnan(iou_matrix)] = -np.inf

        # Use Hungarian algorithm to find optimal assignment
        pred_indices, gt_indices = linear_sum_assignment(
            iou_matrix, maximize=True
        )

        # Mark true positives and false positives based on optimal assignment
        for pred_idx, gt_idx in zip(pred_indices, gt_indices, strict=True):
            if iou_matrix[pred_idx, gt_idx] > iou_threshold:
                true_positives[pred_idx] = True
                matched_gts[gt_idx] = True
                true_positives_iou[pred_idx] = iou_matrix[pred_idx, gt_idx]
            else:
                false_positives[pred_idx] = True

        # Mark unmatched predictions as false positives
        false_positives[~true_positives] = True

        # Mark unmatched ground truth as missed detections
        missed_detections[~matched_gts] = True

    elif len(pred_bboxes) == 0 and len(gt_bboxes) > 0:
        # No predictions, all ground truth are missed
        missed_detections[:] = True
    elif len(pred_bboxes) > 0 and len(gt_bboxes) == 0:
        # No ground truth, all predictions are false positives
        false_positives[:] = True

    # Pad tp, fp for pred_bboxes with False
    tp_fp_pred_bboxes_padded: tuple[np.ndarray, ...] = ()
    for output in [true_positives, false_positives]:
        output_padded = np.pad(
            output,
            (0, n_pred_bboxes_padded - len(output)),
            mode="constant",
            constant_values=False,
        )
        tp_fp_pred_bboxes_padded += (output_padded,)

    # Pad true_positives_iou for pred_bboxes with nan
    true_positives_iou_padded = np.pad(
        true_positives_iou,
        (0, n_pred_bboxes_padded - len(true_positives_iou)),
        mode="constant",
        constant_values=np.nan,
    )

    # Pad results for gt_bboxes with False
    missed_detections_padded = np.pad(
        missed_detections,
        (0, n_gt_bboxes_padded - len(missed_detections)),
        mode="constant",
        constant_values=False,
    )
    return tp_fp_pred_bboxes_padded + (
        missed_detections_padded,
        true_positives_iou_padded,
    )


def compute_precision_recall_ds(
    pred_bboxes_ds: xr.Dataset,
    gt_bboxes_ds: xr.Dataset,
    iou_threshold: float,
) -> tuple[xr.Dataset, xr.Dataset]:
    """Compute precision and recall per image."""
    # Compute true positives, false positives, and missed detections
    pred_bboxes_ds, gt_bboxes_ds = evaluate_detections_hungarian_ds(
        pred_bboxes_ds=pred_bboxes_ds,
        gt_bboxes_ds=gt_bboxes_ds,
        iou_threshold=iou_threshold,
    )

    # Compute precision and recall per image
    precision_per_img = pred_bboxes_ds.tp.sum(dim="id") / (
        pred_bboxes_ds.tp.sum(dim="id") + pred_bboxes_ds.fp.sum(dim="id")
    )
    recall_per_img = pred_bboxes_ds.tp.sum(dim="id") / (
        pred_bboxes_ds.tp.sum(dim="id") + gt_bboxes_ds.md.sum(dim="id")
    )

    # Add to datasets
    pred_bboxes_ds["precision"] = precision_per_img
    pred_bboxes_ds["recall"] = recall_per_img

    return pred_bboxes_ds, gt_bboxes_ds


def _add_bboxes_min_max_corners(ds):
    """Add xy_min and xy_max arrays to ds.

    # Compare to torchvision.ops.box_convert in testing?
    box_convert(
        torch.from_numpy(np.c_[ds.position.T, ds.shape.T]),
        in_fmt="cxcywh",
        out_fmt="xyxy",
    )
    """
    ds["xy_min"] = ds.position - 0.5 * ds.shape
    ds["xy_max"] = ds.position + 0.5 * ds.shape
    return ds
