"""Utilities for evaluating detectors."""

import numpy as np
import torch
import torchvision.ops as ops
from scipy.optimize import linear_sum_assignment


def evaluate_detections_hungarian(
    pred_bboxes: np.ndarray, gt_bboxes: np.ndarray, iou_threshold: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute true positives, false positives, and missed detections.

    Uses Hungarian algorithm for matching.

    Parameters
    ----------
    pred_bboxes : list
        A list of prediction bounding boxes with the first four columns being
        [x1, y1, x2, y2]
    gt_bboxes : list
        A list of ground truth bounding boxes with the first four columns being
        [x1, y1, x2, y2]
    iou_threshold : float
        IoU threshold for considering a detection as true positive

    Returns
    -------
    tuple
        A tuple of three boolean arrays:
        - true_positives: True for each predicted bbox that is a true positive
        - false_positives: True for each predicted bbox that is a false
        positive
        - missed_detections: True for each ground truth bbox that is missed

    """
    # Initialize output arrays
    true_positives = np.zeros(len(pred_bboxes), dtype=bool)
    false_positives = np.zeros(len(pred_bboxes), dtype=bool)
    matched_gts = np.zeros(len(gt_bboxes), dtype=bool)
    missed_detections = np.zeros(len(gt_bboxes), dtype=bool)  # unmatched gts

    # cast as a tensor if not already
    if not isinstance(pred_bboxes, torch.Tensor):
        pred_bboxes = torch.tensor(pred_bboxes, dtype=torch.float32)
    if not isinstance(gt_bboxes, torch.Tensor):
        gt_bboxes = torch.tensor(gt_bboxes, dtype=torch.float32)

    if len(pred_bboxes) > 0 and len(gt_bboxes) > 0:
        # Compute IoU matrix (pred_bboxes x gt_bboxes)
        iou_matrix = ops.box_iou(pred_bboxes[:, :4], gt_bboxes).cpu().numpy()

        # Use Hungarian algorithm to find optimal assignment
        pred_indices, gt_indices = linear_sum_assignment(
            iou_matrix, maximize=True
        )

        # Mark true positives and false positives based on optimal assignment
        for pred_idx, gt_idx in zip(pred_indices, gt_indices, strict=True):
            if iou_matrix[pred_idx, gt_idx] > iou_threshold:
                true_positives[pred_idx] = True
                matched_gts[gt_idx] = True
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

    return true_positives, false_positives, missed_detections
