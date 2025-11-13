"""Wrappers around ensemble-boxes fusion functions."""
import numpy as np
from ensemble_boxes import weighted_boxes_fusion


def weighted_boxes_fusion_in_pixels(
    image_height_width: tuple[int, int],
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    iou_thr: float,
    skip_box_thr: float,
):
    """Fuse bboxes for a single image and return in pixels."""
    # Normalize boxes using image shape
    image_height, image_width = image_height_width
    boxes_list = [
        boxes
        / np.array([image_width, image_height, image_width, image_height])
        if len(boxes) > 0
        else boxes
        for boxes in boxes_list
    ]

    # Apply WBF
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )

    # Denormalize boxes
    # Format of returned bboxes is x1y1x2y2 in pixels like fasterrcnn
    fused_boxes = fused_boxes * np.array(
        [image_width, image_height, image_width, image_height]
    )

    return fused_boxes, fused_scores, fused_labels