"""Run detection on a Pytorch dataset and export results as a movement dataset.

A script to run detection only (no tracking) on a Pytorch dataset and
export the results in a format that can be loaded in movement napari widget.
"""

# %%
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.ops as ops
import torchvision.transforms.v2 as transforms
import xarray as xr
from mlflow.tracking import MlflowClient
from scipy.optimize import linear_sum_assignment
from torch.utils.data import random_split
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set xarray options
xr.set_options(display_expand_attrs=False)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Detection evaluation function


def evaluate_detections(pred_bboxes, gt_bboxes, iou_threshold=0.5):
    """Evaluate detection performance using IoU-based matching.

    Parameters
    ----------
    pred_bboxes : np.ndarray
        Array of predicted bounding boxes with columns [x1, y1, x2, y2, confidence]
    gt_bboxes : np.ndarray
        Array of ground truth bounding boxes with columns [x1, y1, x2, y2]
    iou_threshold : float, optional
        IoU threshold for considering a detection as true positive, default 0.5

    Returns
    -------
    tuple
        (true_positives, false_positives, missed_detections) where each is a boolean array
        - true_positives: column vector with True for each predicted bbox that is a true positive
        - false_positives: column vector with True for each predicted bbox that is a false positive
        - missed_detections: column vector with True for each ground truth bbox that is missed

    """
    # Initialize output arrays
    true_positives = np.zeros(len(pred_bboxes), dtype=bool)
    false_positives = np.zeros(len(pred_bboxes), dtype=bool)
    missed_detections = np.zeros(len(gt_bboxes), dtype=bool)

    if len(pred_bboxes) > 0 and len(gt_bboxes) > 0:
        # Sort predictions by confidence (descending)
        sorted_indices = np.argsort(pred_bboxes[:, 4])[::-1]
        pred_bboxes_sorted = pred_bboxes[sorted_indices]

        # Track which ground truth boxes have been matched
        gt_matched = np.zeros(len(gt_bboxes), dtype=bool)

        # For each prediction, find the best matching ground truth
        for i, pred_bbox in enumerate(pred_bboxes_sorted):
            best_iou = 0
            best_gt_idx = -1

            # Calculate IoU with all unmatched ground truth boxes
            for j, gt_bbox in enumerate(gt_bboxes):
                if gt_matched[j]:
                    continue

                # Calculate IoU using torchvision.ops.box_iou
                pred_tensor = torch.tensor(
                    pred_bbox[:4], dtype=torch.float32
                ).unsqueeze(0)
                gt_tensor = torch.tensor(
                    gt_bbox, dtype=torch.float32
                ).unsqueeze(0)
                iou = ops.box_iou(pred_tensor, gt_tensor).item()

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = j

            # Determine if this prediction is a true positive or false positive
            pred_idx_in_original = sorted_indices[i]

            if best_iou >= iou_threshold and best_gt_idx >= 0:
                # True positive
                true_positives[pred_idx_in_original] = True
                gt_matched[best_gt_idx] = True
            else:
                # False positive
                false_positives[pred_idx_in_original] = True

        # Mark unmatched ground truth as missed detections
        missed_detections = ~gt_matched

    elif len(pred_bboxes) == 0 and len(gt_bboxes) > 0:
        # No predictions, all ground truth are missed
        missed_detections[:] = True
    elif len(pred_bboxes) > 0 and len(gt_bboxes) == 0:
        # No ground truth, all predictions are false positives
        false_positives[:] = True

    return true_positives, false_positives, missed_detections


def evaluate_detections_hungarian(
    pred_bboxes: np.ndarray, gt_bboxes: np.ndarray, iou_threshold: float
) -> dict:
    """Evaluate detection performance using Hungarian algorithm for matching.

    Parameters
    ----------
    pred_bboxes : list
        A list of prediction bounding boxes with columns [x1, y1, x2, y2, confidence]
    gt_bboxes : list
        A list of ground truth bounding boxes with columns [x1, y1, x2, y2]
    iou_threshold : float
        IoU threshold for considering a detection as true positive

    Returns
    -------
    tuple
        (true_positives, false_positives, missed_detections) where each is a boolean array
        - true_positives: column vector with True for each predicted bbox that is a true positive
        - false_positives: column vector with True for each predicted bbox that is a false positive
        - missed_detections: column vector with True for each ground truth bbox that is missed

    """
    # Initialize output arrays
    true_positives = np.zeros(len(pred_bboxes), dtype=bool)
    false_positives = np.zeros(len(pred_bboxes), dtype=bool)
    matched_gts = np.zeros(len(gt_bboxes), dtype=bool)
    missed_detections = np.zeros(len(gt_bboxes), dtype=bool)  # unmatched gts

    if len(pred_bboxes) > 0 and len(gt_bboxes) > 0:
        # Compute IoU matrix (pred_bboxes x gt_bboxes)
        iou_matrix = (
            ops.box_iou(
                torch.tensor(pred_bboxes[:, :4], dtype=torch.float32),
                torch.tensor(gt_bboxes, dtype=torch.float32),
            )
            .cpu()
            .numpy()
        )

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

    # Return sum as a dict
    return true_positives, false_positives, missed_detections


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
dataset_dir = Path("/home/sminano/swc/project_crabs/data/sep2023-full")

trained_model_path = Path(
    "/home/sminano/swc/project_crabs/ml-runs/617393114420881798/f348d9d196934073bece1b877cbc4d38/checkpoints/last.ckpt"
)

trained_model_mlflow_params_path = Path(
    "/home/sminano/swc/project_crabs/ml-runs/617393114420881798/f348d9d196934073bece1b877cbc4d38/params"
)  # for config


# to save output frames and detections
output_parent_dir = Path("/home/sminano/swc/project_ethology")

flag_save_frames = False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set default device: CUDA if available, otherwise mps, otherwise CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Retrieve model config and CLI args from mlflow


def read_mlflow_params(
    trained_model_path: str, tracking_uri: str = None
) -> dict:
    """Read parameters for a specific MLflow run."""
    # Create MLflow client
    mlruns_path = str(Path(trained_model_path).parents[3])
    client = MlflowClient(tracking_uri=mlruns_path)

    # Get the run
    runID = Path(trained_model_path).parents[1].stem
    run = client.get_run(runID)

    # Access parameters
    params = run.data.params
    params["run_name"] = run.info.run_name

    return params


mlflow_params = read_mlflow_params(trained_model_path)
config = {
    k.removeprefix("config/"): ast.literal_eval(v)
    for k, v in mlflow_params.items()
    if k.startswith("config/")
}


def safe_eval_string(s):
    """Try to evaluate a string as a literal, otherwise return as-is."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # return as-is if not a valid literal
        return s


cli_args = {
    k.removeprefix("cli_args/"): safe_eval_string(v)
    for k, v in mlflow_params.items()
    if k.startswith("cli_args/")
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load model

# Load structure
model = fasterrcnn_resnet50_fpn_v2(
    weights=None,
    weights_backbone=None,
    num_classes=config["num_classes"],
)

# Read state dict
state_dict = torch.load(trained_model_path)
state_dict_model = {
    k.lstrip("model."): v
    for k, v in state_dict["state_dict"].items()
    if k.startswith("model.")
}

# Load weights into model and set to evaluation mode
model.load_state_dict(state_dict_model)
model.eval()
model.to(device)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define transforms to apply to input frames
inference_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

# Sanitize bounding boxes?

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Build Pytorch dataset
seed_n = cli_args["seed_n"]
annotations_filename = Path(cli_args["annotation_files"][0]).name

# create "default" COCO dataset
dataset_coco = CocoDetection(
    Path(dataset_dir) / "frames",
    Path(dataset_dir) / "annotations" / annotations_filename,
    transforms=inference_transforms,
)

# wrap dataset for transforms v2
dataset_transformed = wrap_dataset_for_transforms_v2(dataset_coco)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Split dataset
# def _collate_fn(self, batch: tuple) -> tuple:
#     """Collate function used for dataloaders.

#     A custom function is needed for detection
#     because the number of bounding boxes varies
#     between images of the same batch.
#     See https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html#data-loading-and-training-loop

#     Parameters
#     ----------
#     batch : tuple
#         a tuple of 2 tuples, the first one holding all images in the batch,
#         and the second one holding the corresponding annotations.

#     Returns
#     -------
#     tuple
#         a tuple of length = batch size, made up of (image, annotations)
#         tuples.

#     """
#     return tuple(zip(*batch))


# Split data into train and test-val sets
rng_train_split = torch.Generator().manual_seed(seed_n)
rng_val_split = torch.Generator().manual_seed(seed_n)

train_dataset, test_val_dataset = random_split(
    dataset_transformed,
    [config["train_fraction"], 1 - config["train_fraction"]],
    generator=rng_train_split,
)

# Split test/val sets from the remainder
test_dataset, val_dataset = random_split(
    test_val_dataset,
    [
        1 - config["val_over_test_fraction"],
        config["val_over_test_fraction"],
    ],
    generator=rng_val_split,
)

print(f"Seed: {seed_n}")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run detection on validation set

# TODO: use dataloader for efficiency?
detections_per_validation_sample = {}

for val_idx, (image, annotations) in enumerate(val_dataset):
    # Apply transforms to frame and place tensor on device
    image_tensor = inference_transforms(image).to(device)[None]

    # Put annotations in same device as image
    annotations["boxes"] = annotations["boxes"].to(device)
    annotations["labels"] = annotations["labels"].to(device)

    # Run detection
    with torch.no_grad():
        # use [0] to select the one image in the batch
        # Returns: dictionary with data of the predicted bounding boxes.
        # The keys are: "boxes", "scores", and "labels". The labels
        # refer to the class of the object detected, and not its ID.
        detections_dict = model(image_tensor)[0]

    # Add to dict
    bboxes_xyxy = detections_dict["boxes"].cpu().numpy()

    detections_per_validation_sample[val_idx] = {
        "bbox_xyxy": bboxes_xyxy,
        "bbox_centroids": (bboxes_xyxy[:, 0:2] + bboxes_xyxy[:, 2:4]) / 2,
        "bbox_shapes": bboxes_xyxy[:, 2:4] - bboxes_xyxy[:, 0:2],
        "bbox_confidences": detections_dict["scores"].cpu().numpy(),
        "bbox_labels": detections_dict["labels"].cpu().numpy(),
    }


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Evaluate detections using Hungarian algorithm and create dataframes

iou_threshold = 0.1

# Collect all data efficiently
list_pred_subtables = []
list_gt_subtables = []


# Loop over validation set
for val_idx, (image, annotations) in enumerate(val_dataset):
    # Get predictions for this image
    pred_dict = detections_per_validation_sample[val_idx]
    pred_bboxes = np.column_stack(
        [pred_dict["bbox_xyxy"], pred_dict["bbox_confidences"]]
    )

    # Get ground truth
    gt_bboxes = annotations["boxes"].cpu().numpy()

    # Evaluate detections
    tp, fp, md = evaluate_detections_hungarian(
        pred_bboxes, gt_bboxes, iou_threshold
    )

    # Calculate bboxes areas
    pred_bboxes_width = pred_bboxes[:, 2] - pred_bboxes[:, 0]
    pred_bboxes_height = pred_bboxes[:, 3] - pred_bboxes[:, 1]
    pred_areas = pred_bboxes_width * pred_bboxes_height

    gt_bboxes_width = gt_bboxes[:, 2] - gt_bboxes[:, 0]
    gt_bboxes_height = gt_bboxes[:, 3] - gt_bboxes[:, 1]
    gt_areas = gt_bboxes_width * gt_bboxes_height

    # Create prediction subtable
    pred_data = {
        "prediction_ID": [
            f"pred_{val_idx}_{i}" for i in range(len(pred_bboxes))
        ],
        "image_ID": annotations["image_id"],
        "confidence": pred_dict["bbox_confidences"],
        "TP": tp,
        "FP": fp,
        "bbox_area": pred_areas,
    }
    list_pred_subtables.append(pd.DataFrame(pred_data))

    # Create ground truth subtable
    gt_data = {
        "gt_annotation_ID": [
            f"gt_{val_idx}_{i}" for i in range(len(gt_bboxes))
        ],
        "image_ID": annotations["image_id"],
        "missed_detection": md,
        "bbox_area": gt_areas,
    }
    list_gt_subtables.append(pd.DataFrame(gt_data))

# Concatenate all dataframes
predictions_df = pd.concat(list_pred_subtables, ignore_index=True)
gt_annotations_df = pd.concat(list_gt_subtables, ignore_index=True)


# %%
gt_area_percentiles = np.percentile(
    gt_annotations_df["bbox_area"], np.arange(0, 105, 5)
)

bin_labels = [
    f"{gt_area_percentiles[i]:.0f}-{gt_area_percentiles[i + 1]:.0f}"
    for i in range(gt_area_percentiles.shape[0] - 1)
]


predictions_df["area_bins"] = pd.cut(
    predictions_df["bbox_area"],
    bins=gt_area_percentiles,  # same bins for predictions and gt
    labels=bin_labels,
    include_lowest=True,
    right=False,
)

gt_annotations_df["area_bins"] = pd.cut(
    gt_annotations_df["bbox_area"],
    bins=gt_area_percentiles,  # same bins for predictions and gt
    labels=bin_labels,
    include_lowest=True,
    right=False,
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Count detections in each bin
# Is GT really that balanced??
predictions_per_area_bin = (
    predictions_df["area_bins"].value_counts().sort_index()
)
gt_per_area_bin = gt_annotations_df["area_bins"].value_counts().sort_index()

comparison_df = pd.DataFrame(
    {"Predictions": predictions_per_area_bin, "Ground Truth": gt_per_area_bin}
)

# Plot as bar chart
plt.figure(figsize=(10, 6))
comparison_df.plot(
    kind="bar",
    figsize=(12, 6),
    color=["skyblue", "lightcoral"],
    stacked=False,
)
plt.title("Detection Counts by Area Bins Validation Set")
plt.xlabel("Area Range (pixels^2)")
plt.ylabel("Number of Detections")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Count true positives per bin

true_positives_counts = pd.DataFrame(
    {
        "Predictions": predictions_per_area_bin,
        # "Ground Truth": gt_per_area_bin,
        "True Positives": predictions_df.loc[predictions_df["TP"], "area_bins"]
        .value_counts()
        .sort_index(),
    }
)

# Plot as bar chart
true_positives_counts.plot(
    kind="bar",
    figsize=(12, 6),
    color=["skyblue", "blue"],
    stacked=False,
)
plt.title("Counts per Area Bin Validation Set")
plt.xlabel("Bbox area (pixels^2)")
plt.ylabel("Number of Detections")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Count missed detections per bin

missed_detections_counts = pd.DataFrame(
    {
        "Ground Truth": gt_per_area_bin,
        "Matched Ground Truth": gt_annotations_df.loc[
            ~gt_annotations_df["missed_detection"], "area_bins"
        ]
        .value_counts()
        .sort_index(),
    }
)

# Plot as bar chart
missed_detections_counts.plot(
    kind="bar",
    figsize=(12, 6),
    color=["lightcoral", "green"],
    stacked=False,
)
plt.title("Counts per Area Bin Validation Set")
plt.xlabel("Area Range (pixels^2)")
plt.ylabel("Number of Detections")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# %%%%%%%%%%%
# Image id histogram

detections_per_image_id = pd.DataFrame(
    {
        "Predictions": predictions_df.groupby("image_ID").count()[
            "prediction_ID"
        ],
        "Ground Truth": gt_annotations_df.groupby("image_ID").count()[
            "gt_annotation_ID"
        ],
        "True Positives": predictions_df.groupby("image_ID")["TP"].sum(),
    }
)

# Plot as bar chart
plt.figure(figsize=(10, 6))
detections_per_image_id.plot(
    kind="bar",
    figsize=(12, 6),
    color=["skyblue", "lightcoral", "green"],
    stacked=False,
)
plt.title("Detections per Image ID")
plt.xlabel("Image ID")
plt.ylabel("Number of Detections")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
