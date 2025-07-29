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
from ethology.detectors.evaluate import evaluate_detections_hungarian_arrays
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
annotations_dir = Path("/home/sminano/swc/project_ethology/large_annotations")


list_models = [
    "above_0th",
    "above_1st",
    "above_5th",
    "above_10th",
    "above_25th",
    "above_50th",
]
timestamp_ref = "20250717_115247"
predictions_dir = Path(
    "/home/sminano/swc/project_ethology/remove_small_bboxes_inD_output"
)

flag_use_full_gt = True
full_gt_annotations_file = (
    annotations_dir / "VIA_JSON_combined_coco_gen_sorted_imageIDs.json"
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Helper functions


def split_dataset_crab_repo(dataset_coco, seed_n, config):
    """Split dataset like in crabs repo."""
    # Split data into train and test-val sets
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

    return train_dataset, val_dataset, test_dataset


def compute_pred_gt_tables(iou_threshold, ds_predictions, val_dataset):
    list_pred_subtables = []
    list_gt_subtables = []

    # Loop over validation set
    for val_idx, (_img, annotations) in enumerate(val_dataset):
        # Get image_id from validation set
        image_id = annotations["image_id"]

        # Get predictions for this image
        centroids = ds_predictions.centroids.isel(image_id=val_idx).T.values
        shape = ds_predictions.shape.isel(image_id=val_idx).T.values
        confidence = ds_predictions.confidence.isel(image_id=val_idx).T.values
        slc_non_nan = ~np.isnan(centroids).any(axis=1)

        # format predictions as xyxy
        pred_bboxes = np.concatenate(
            [
                centroids[slc_non_nan] - (shape[slc_non_nan] / 2),
                centroids[slc_non_nan] + (shape[slc_non_nan] / 2),
            ],
            axis=1,
        )

        # Get ground truth from input dataset
        gt_bboxes = annotations["boxes"].cpu().numpy()

        # Evaluate detections
        tp, fp, md, _ = evaluate_detections_hungarian_arrays(
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
            "image_ID": image_id,
            "val_batch_idx": val_idx,
            "confidence": confidence[slc_non_nan],
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
            "image_ID": image_id,
            "val_batch_idx": val_idx,
            "missed_detection": md,
            "bbox_diagonal": gt_diagonals,
        }
        list_gt_subtables.append(pd.DataFrame(gt_data))

    # Concatenate all dataframes
    predictions_df = pd.concat(list_pred_subtables, ignore_index=True)
    gt_annotations_df = pd.concat(list_gt_subtables, ignore_index=True)

    return predictions_df, gt_annotations_df


def compute_average_precision_recall_per_image_id(
    predictions_df, gt_annotations_df
):
    precision_recall_df = pd.DataFrame(
        {
            "TP": predictions_df.groupby("image_ID")["TP"].sum(),
            "FP": predictions_df.groupby("image_ID")["FP"].sum(),
            "MD": gt_annotations_df.groupby("image_ID")[
                "missed_detection"
            ].sum(),
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

    # print(precision_recall_df)
    print(f"Average precision: {precision_recall_df['precision'].mean()}")
    print(f"Average recall: {precision_recall_df['recall'].mean()}")

    return precision_recall_df


def discretize_based_on_bbox_diagonal(
    predictions_df,
    gt_annotations_df,
    gt_diagonal_percentile_values,
    bin_labels,
):
    predictions_df["diagonal_bins"] = pd.cut(
        predictions_df["bbox_diagonal"],
        bins=gt_diagonal_percentile_values,  # same bins for predictions and gt
        labels=bin_labels,
        include_lowest=False,
        right=True,
    )

    gt_annotations_df["diagonal_bins"] = pd.cut(
        gt_annotations_df["bbox_diagonal"],
        bins=gt_diagonal_percentile_values,  # same bins for predictions and gt
        labels=bin_labels,
        include_lowest=False,
        right=True,
    )

    return predictions_df, gt_annotations_df


def plot_true_positives_per_bin(
    predictions_df, predictions_per_diagonal_bin, model_key
):
    """Plot true positives per diagonal bin"""
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
    ax.set_title(f"model trained on annotations {model_key} percentile")
    ax.set_xlabel("diagonal (pixels)")
    ax.set_ylabel("count")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0.0, 425)
    ax.grid(True, alpha=0.3)

    # add line plot for precision on right y-axis
    ax2 = ax.twinx()
    ax2.plot(
        range(len(true_positives_counts)),
        true_positives_counts["precision"],
        color="red",
        marker="o",
        label="precision",
        linewidth=2,
    )
    ax2.set_ylabel("precision", color="red")
    ax2.tick_params(axis="y", labelcolor="red")
    ax2.set_ylim(0.0, 1.00)  # Precision is between 0 and 1

    # add reference line at 0.96
    ax2.axhline(y=0.96, color="red", linestyle="--", linewidth=1)

    # put legend on left
    ax.legend(loc="upper left")
    plt.tight_layout()
    plt.show()


def plot_missed_detections_per_bin(
    gt_annotations_df, gt_per_diagonal_bin, model_key
):
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
    missed_detections_counts.loc[
        :, ["Ground Truth", "Matched Ground Truth"]
    ].plot(
        kind="bar",
        ax=ax,
        figsize=(12, 6),
        color=["lightcoral", "green"],
        stacked=False,
    )
    ax.set_title(f"model trained on annotations {model_key} percentile")
    ax.set_xlabel("Diagonal (pixels)")
    ax.set_ylabel("Number of Detections")
    ax.tick_params(axis="x", rotation=45)
    ax.set_ylim(0.0, 400)
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

    # add reference line at 0.85
    ax2.axhline(y=0.85, color="blue", linestyle="--", linewidth=1)

    plt.tight_layout()
    plt.show()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute bins using full GT annotations
# We bin the size of the bbox diagonal

coco_full_gt = COCO(str(full_gt_annotations_file))

# compute diagonals for each gt annotation
gt_bboxes_diagonals = np.array(
    [
        np.sqrt(
            annot["bbox"][2] ** 2 + annot["bbox"][3] ** 2
        )  # bbox is xywh in COCO
        for annot in coco_full_gt.dataset["annotations"]
    ]
)

# compute percentiles of diagonals
gt_diagonal_percentiles = np.arange(0, 105, 5)
gt_diagonal_percentile_values = np.percentile(
    gt_bboxes_diagonals, gt_diagonal_percentiles
)

# define labels for bins
bin_labels = [
    f"{gt_diagonal_percentile_values[i]:.0f}-{gt_diagonal_percentile_values[i + 1]:.0f}"
    for i in range(gt_diagonal_percentile_values.shape[0] - 1)
]

print(gt_diagonal_percentiles)
print(gt_diagonal_percentile_values)
print(bin_labels)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot histogram of gt bboxes diagonals with bins

fig, ax1 = plt.subplots(figsize=(10, 6))

# histogram
ax1.hist(gt_bboxes_diagonals, bins=100, color="skyblue")

# add vertical lines the bin labels
for i, bin_label in enumerate(bin_labels):
    ax1.axvline(x=gt_diagonal_percentile_values[i], color="red", linestyle="-")
    ax1.text(
        gt_diagonal_percentile_values[i],
        2125,
        gt_diagonal_percentiles[i],
        color="red",
        ha="left",
        va="bottom",
        rotation=45,
        fontsize=6.5,
    )

# ax1.set_title("GT bboxes diagonals")
ax1.set_xlabel("diagonal (pixels)")
ax1.set_ylabel("count")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Evaluate all models


inference_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

default_dataset_coco = create_coco_dataset(
    images_dir=Path(dataset_dir) / "frames",
    annotations_file=full_gt_annotations_file,  # annotations_dir / annotations_filename,
    composed_transform=inference_transforms,
)

iou_threshold = 0.1

for model_key in list_models:
    # ---------------------------------------
    # Load predictions
    ds_predictions = xr.open_dataset(
        predictions_dir
        / f"{model_key}_detections_val_set_seed_42_{timestamp_ref}.nc"
    )

    # ---------------------------------------
    # Define GT annotations

    trained_model_path = ds_predictions.attrs["model_path"]
    mlflow_params = read_mlflow_params(trained_model_path)
    config = read_config_from_mlflow_params(mlflow_params)
    cli_args = read_cli_args_from_mlflow_params(mlflow_params)

    # Create COCO dataset
    # Fix for model trained on all annotations
    # (VIA_JSON_combined_coco_gen has different image IDs than the rest)
    if (
        Path(cli_args["annotation_files"][0]).name
        == "VIA_JSON_combined_coco_gen.json"
    ):
        dataset_coco = create_coco_dataset(
            images_dir=Path(dataset_dir) / "frames",
            annotations_file=annotations_dir
            / "VIA_JSON_combined_coco_gen.json",
            composed_transform=inference_transforms,
        )
    else:
        dataset_coco = default_dataset_coco

    # Split dataset like in crabs repo
    train_dataset, val_dataset, test_dataset = split_dataset_crab_repo(
        dataset_coco,
        seed_n=cli_args["seed_n"],
        config=config,
    )

    # ---------------------------------------
    # Evaluate detections using Hungarian algorithm and create dataframes

    predictions_df, gt_annotations_df = compute_pred_gt_tables(
        iou_threshold, ds_predictions, val_dataset
    )

    # ---------------------------------------
    # Check average precision and recall on validation set
    precision_recall_df = compute_average_precision_recall_per_image_id(
        predictions_df, gt_annotations_df
    )

    # all annotations:
    # Average precision: 0.9456786718983294
    # Average recall: 0.8494677009613534

    # ---------------------------------------
    # Discretize annotations based on bbox diagonal
    predictions_df, gt_annotations_df = discretize_based_on_bbox_diagonal(
        predictions_df,
        gt_annotations_df,
        gt_diagonal_percentile_values,
        bin_labels,
    )

    # ---------------------------------------
    # Plot boxes in each diagonal bin in validation set
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
    plt.ylim(0.0, 400)
    plt.title(f"model trained on annotations {model_key} percentile")
    plt.xlabel("diagonal (pixels)")
    plt.ylabel("count")
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # ---------------------------------------
    # Plot true positives per bin
    plot_true_positives_per_bin(
        predictions_df, predictions_per_diagonal_bin, model_key
    )

    # ---------------------------------------
    # Plot missed detections per bin
    plot_missed_detections_per_bin(
        gt_annotations_df, gt_per_diagonal_bin, model_key
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Discretize predictions based on confidence
bin_edges = np.arange(0, 1.01, 0.05)
predictions_df["confidence_bins"] = pd.cut(
    predictions_df["confidence"],
    bins=bin_edges,
)

precision_per_confidence_bin = predictions_df.groupby(
    "confidence_bins", observed=False
)["TP"].sum()
total_detections_per_confidence_bin = (
    predictions_df["confidence_bins"].value_counts().sort_index()
)

calibration_df = pd.DataFrame(
    {
        "precision": precision_per_confidence_bin
        / total_detections_per_confidence_bin,
        "total_detections": total_detections_per_confidence_bin,
        "TP": precision_per_confidence_bin,
    }
)

# Plot as bar chart
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
calibration_df["precision"].plot(
    kind="bar",
    figsize=(12, 6),
    color=["skyblue"],
    ax=ax,
)

ax.plot(
    np.arange(len(calibration_df)),  # bin indices
    (bin_edges[:-1] + bin_edges[1:]) / 2,  # perfect calibration
    color="red",
    linewidth=2,
    marker="o",
    label="Perfect calibration",
)

ax.set_title(
    f"{model_key} - calibration curve (n={precision_per_confidence_bin.sum()})"
)
ax.set_xlabel("confidence")
ax.set_ylabel("Precision")
ax.tick_params(axis="x", rotation=45)
ax.grid(True, alpha=0.3)

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
# %matplotlib widget
# %%
