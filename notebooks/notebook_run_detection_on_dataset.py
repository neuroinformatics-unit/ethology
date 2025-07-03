"""Run detection on a Pytorch dataset and export results as a movement dataset.

A script to run detection only (no tracking) on a Pytorch dataset and
export the results in a format that can be loaded in movement napari widget.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
import xarray as xr
from pycocotools.coco import COCO
from torch.utils.data import random_split

from ethology.datasets.create import create_coco_dataset
from ethology.detectors.evaluate import evaluate_detections_hungarian
from ethology.detectors.inference import run_detector_on_dataset
from ethology.detectors.load import load_fasterrcnn_resnet50_fpn_v2
from ethology.mlflow import (
    read_cli_args_from_mlflow_params,
    read_config_from_mlflow_params,
    read_mlflow_params,
)

# Set xarray options
xr.set_options(display_expand_attrs=False)

# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data - in domain
dataset_dir = Path("/home/sminano/swc/project_crabs/data/sep2023-full")


experiment_ID = "617393114420881798"
ml_runs_experiment_dir = (
    Path("/home/sminano/swc/project_crabs/ml-runs") / experiment_ID
)
annotations_dir = Path("/home/sminano/swc/project_ethology/large_annotations")


# percentile is of bbox diagonal!
models_dict = {
    "above_0th_percentile_seed_42": (
        ml_runs_experiment_dir
        / "f348d9d196934073bece1b877cbc4d38"
        / "checkpoints"
        / "last.ckpt"
    ),
    "above_5th_percentile_seed_42": (
        ml_runs_experiment_dir
        / "e72e53b23df142ae859dd590798b4162"
        / "checkpoints"
        / "last.ckpt"
    ),
}


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%

full_gt_annotations_file = annotations_dir / "VIA_JSON_combined_coco_gen.json"
coco_full_gt = COCO(str(full_gt_annotations_file))

# compute diagonal percentiles for full gt
gt_bboxes_diagonals = [
    np.sqrt(
        annot["bbox"][2] ** 2 + annot["bbox"][3] ** 2
    )  # bbox is xywh in COCO
    for annot in coco_full_gt.dataset["annotations"]
]
gt_diagonal_percentiles = np.percentile(
    gt_bboxes_diagonals, np.arange(0, 105, 5)
)


bin_labels = [
    f"{gt_diagonal_percentiles[i]:.0f}-{gt_diagonal_percentiles[i + 1]:.0f}"
    for i in range(gt_diagonal_percentiles.shape[0] - 1)
]

print(bin_labels)

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

trained_model_path = str(models_dict["above_5th_percentile_seed_42"])

mlflow_params = read_mlflow_params(trained_model_path)
config = read_config_from_mlflow_params(mlflow_params)
cli_args = read_cli_args_from_mlflow_params(mlflow_params)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load model

model = load_fasterrcnn_resnet50_fpn_v2(
    trained_model_path,
    num_classes=config["num_classes"],
    device=device,
)

# Set to evaluation mode
model.eval()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create COCO dataset
annotations_filename = Path(cli_args["annotation_files"][0]).name
print(annotations_filename)

inference_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)


dataset_coco = create_coco_dataset(
    images_dir=Path(dataset_dir) / "frames",
    annotations_file=annotations_dir / annotations_filename,
    composed_transform=inference_transforms,
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Split dataset like in crabs repo

# Split data into train and test-val sets
seed_n = cli_args["seed_n"]
rng_train_split = torch.Generator().manual_seed(seed_n)
rng_val_split = torch.Generator().manual_seed(seed_n)

# Split train and test-val sets
train_dataset, test_val_dataset = random_split(
    dataset_coco,
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
print(f"Number of training samples: {len(train_dataset)}")  # images
print(f"Number of validation samples: {len(val_dataset)}")  # images
print(f"Number of test samples: {len(test_dataset)}")  # images

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create dataloader

# dataloader = torch.utils.data.DataLoader(
#     val_dataset,
#     batch_size=1,
#     shuffle=True,
# )

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run detection on validation set
detections_dict_per_sample = run_detector_on_dataset(
    model=model,
    dataset=val_dataset,
    device=device,
)

# reshape
detections_per_validation_sample = {}
for val_idx in range(len(val_dataset)):
    detections_dict = detections_dict_per_sample[val_idx]
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

    # Calculate bboxes diagonals
    pred_bboxes_width = pred_bboxes[:, 2] - pred_bboxes[:, 0]
    pred_bboxes_height = pred_bboxes[:, 3] - pred_bboxes[:, 1]
    pred_diagonals = np.sqrt(pred_bboxes_width**2 + pred_bboxes_height**2)

    gt_bboxes_width = gt_bboxes[:, 2] - gt_bboxes[:, 0]
    gt_bboxes_height = gt_bboxes[:, 3] - gt_bboxes[:, 1]
    gt_diagonals = np.sqrt(gt_bboxes_width**2 + gt_bboxes_height**2)

    # Create prediction subtable
    pred_data = {
        "prediction_ID": [
            f"pred_{val_idx}_{i}" for i in range(len(pred_bboxes))
        ],
        "image_ID": annotations["image_id"],
        "val_batch_idx": val_idx,
        "confidence": pred_dict["bbox_confidences"],
        "TP": tp,
        "FP": fp,
        "bbox_diagonal": pred_diagonals,
    }
    list_pred_subtables.append(pd.DataFrame(pred_data))

    # Create ground truth subtable
    gt_data = {
        "gt_annotation_ID": [
            f"gt_{val_idx}_{i}" for i in range(len(gt_bboxes))
        ],
        "image_ID": annotations["image_id"],
        "val_batch_idx": val_idx,
        "missed_detection": md,
        "bbox_diagonal": gt_diagonals,
    }
    list_gt_subtables.append(pd.DataFrame(gt_data))

# Concatenate all dataframes
predictions_df = pd.concat(list_pred_subtables, ignore_index=True)
gt_annotations_df = pd.concat(list_gt_subtables, ignore_index=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check average precision and recall on validation set


precision_recall_df = pd.DataFrame(
    {
        "TP": predictions_df.groupby("image_ID")["TP"].sum(),
        "FP": predictions_df.groupby("image_ID")["FP"].sum(),
        "MD": gt_annotations_df.groupby("image_ID")["missed_detection"].sum(),
        "GT": gt_annotations_df.groupby("image_ID").count()[
            "gt_annotation_ID"
        ],
        "val_batch_idx": predictions_df.groupby("image_ID")[
            "val_batch_idx"
        ].first(),
    }
)

# sort by val_batch_idx
precision_recall_df = precision_recall_df.sort_values(by="val_batch_idx")
precision_recall_df = precision_recall_df.reset_index()

precision_recall_df["precision"] = precision_recall_df["TP"] / (
    precision_recall_df["TP"] + precision_recall_df["FP"]
)
precision_recall_df["recall"] = (
    precision_recall_df["TP"] / precision_recall_df["GT"]
)

print(precision_recall_df)
print(f"Average precision: {precision_recall_df['precision'].mean()}")
print(f"Average recall: {precision_recall_df['recall'].mean()}")


# all annotations:
# Average precision: 0.9456786718983294
# Average recall: 0.8494677009613534

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Discretize annotations based on diagonal


predictions_df["diagonal_bins"] = pd.cut(
    predictions_df["bbox_diagonal"],
    bins=gt_diagonal_percentiles,  # same bins for predictions and gt
    labels=bin_labels,
    include_lowest=True,
    right=False,
)

gt_annotations_df["diagonal_bins"] = pd.cut(
    gt_annotations_df["bbox_diagonal"],
    bins=gt_diagonal_percentiles,  # same bins for predictions and gt
    labels=bin_labels,
    include_lowest=True,
    right=False,
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Count detections in each diagonal bin
# Is GT really that balanced??
predictions_per_diagonal_bin = (
    predictions_df["diagonal_bins"].value_counts().sort_index()
)
gt_per_diagonal_bin = (
    gt_annotations_df["diagonal_bins"].value_counts().sort_index()
)

comparison_df = pd.DataFrame(
    {
        "Predictions": predictions_per_diagonal_bin,
        "Ground Truth": gt_per_diagonal_bin,
    }
)

# Plot as bar chart
plt.figure(figsize=(10, 6))
comparison_df.plot(
    kind="bar",
    figsize=(12, 6),
    color=["skyblue", "lightcoral"],
    stacked=False,
)
plt.title("Detection Counts by Diagonal Bins Validation Set")
plt.xlabel("Diagonal (pixels)")
plt.ylabel("Number of Detections")
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Count true positives per bin

true_positives_counts = pd.DataFrame(
    {
        "Predictions": predictions_per_diagonal_bin,
        "True Positives": predictions_df.loc[
            predictions_df["TP"], "diagonal_bins"
        ]
        .value_counts()
        .sort_index(),
    }
)

true_positives_counts["precision"] = (
    true_positives_counts["True Positives"]
    / true_positives_counts["Predictions"]
)

# Plot as bar chart
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
true_positives_counts.loc[:, ["Predictions", "True Positives"]].plot(
    kind="bar",
    ax=ax,
    figsize=(12, 6),
    color=["skyblue", "blue"],
    stacked=False,
)
ax.set_title("Counts per Diagonal Bin Validation Set")
ax.set_xlabel("Diagonal (pixels)")
ax.set_ylabel("Number of Detections")
ax.tick_params(axis="x", rotation=45)
ax.set_ylim(0.0, 325)
ax.grid(True, alpha=0.3)


# add line plot for precision on right y-axis
ax2 = ax.twinx()
ax2.plot(
    range(len(true_positives_counts)),
    true_positives_counts["precision"],
    color="red",
    marker="o",
    label="Precision",
    linewidth=2,
)
ax2.set_ylabel("Precision", color="red")
ax2.tick_params(axis="y", labelcolor="red")
ax2.set_ylim(0.0, 1.00)  # Precision is between 0 and 1

plt.tight_layout()
plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Count missed detections per bin

missed_detections_counts = pd.DataFrame(
    {
        "Ground Truth": gt_per_diagonal_bin,
        "Matched Ground Truth": gt_annotations_df.loc[
            ~gt_annotations_df["missed_detection"], "diagonal_bins"
        ]
        .value_counts()
        .sort_index(),
    }
)

missed_detections_counts["recall"] = (
    missed_detections_counts["Matched Ground Truth"]
    / missed_detections_counts["Ground Truth"]
)

# Plot as bar chart
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
missed_detections_counts.loc[:, ["Ground Truth", "Matched Ground Truth"]].plot(
    kind="bar",
    ax=ax,
    figsize=(12, 6),
    color=["lightcoral", "green"],
    stacked=False,
)
ax.set_title("Counts per Diagonal Bin Validation Set")
ax.set_xlabel("Diagonal (pixels)")
ax.set_ylabel("Number of Detections")
ax.tick_params(axis="x", rotation=45)
ax.set_ylim(0.0, 325)
ax.grid(True, alpha=0.3)


# add line plot for recall on right y-axis
ax2 = ax.twinx()
ax2.plot(
    range(len(missed_detections_counts)),
    missed_detections_counts["recall"],
    color="blue",
    marker="o",
    linewidth=2,
)
ax2.tick_params(axis="y", labelcolor="blue")
ax2.set_ylabel("Recall", color="blue")
ax2.set_ylim(0.0, 1.00)  # Recall is between 0 and 1

plt.tight_layout()
plt.show()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

# %%
