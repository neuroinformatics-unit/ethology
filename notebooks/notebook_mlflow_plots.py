"""Run detection on a Pytorch dataset and export results as a movement dataset.

A script to run detection only (no tracking) on a Pytorch dataset and
export the results in a format that can be loaded in movement napari widget.
"""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from pycocotools.coco import COCO

# Set xarray options
xr.set_options(display_expand_attrs=False)

# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data - in domain
dataset_dir = Path("/home/sminano/swc/project_crabs/data/sep2023-full")
annotations_dir = Path("/home/sminano/swc/project_ethology/large_annotations")


map_run_slurm_977884_jobID_to_percentile = {
    "0": "0",
    "1": "0",
    "2": "0",
    "3": "1",
    "4": "1",
    "5": "1",
    "6": "5",
    "7": "5",
    "8": "5",
    "9": "10",
    "10": "10",
    "11": "10",
    "12": "25",
    "13": "25",
    "14": "25",
    "15": "50",
    "16": "50",
    "17": "50",
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute bins using full GT annotations
# We bin the size of the bbox diagonal

full_gt_annotations_file = (
    annotations_dir / "VIA_JSON_combined_coco_gen_sorted_imageIDs.json"
)
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

# %%
# Plot histogram of gt bboxes diagonals

fig, ax1 = plt.subplots(figsize=(10, 6))

# histogram
ax1.hist(gt_bboxes_diagonals, bins=100, color="skyblue")

# add vertical lines for a subset of the percentiles
percentile_subset = [gt_diagonal_percentiles[i] for i in [0, 1, 2, 5, 10]]
percentile_subset_values = [
    gt_diagonal_percentile_values[i] for i in [0, 1, 2, 5, 10]
]
for i, percentile in enumerate(percentile_subset_values):
    ax1.axvline(x=percentile, color="red", linestyle="-")
    ax1.text(
        percentile,
        2125,
        f"{percentile_subset[i]}%",
        color="red",
        ha="left",
        va="bottom",
    )

# manually plot 1% percentile
ax1.axvline(
    x=np.percentile(gt_bboxes_diagonals, 1), color="red", linestyle="-"
)
ax1.text(
    np.percentile(gt_bboxes_diagonals, 1),
    2125,
    "1%",
    color="red",
    ha="left",
    va="bottom",
)

# ax1.set_title("GT bboxes diagonals")
ax1.set_xlabel("diagonal (pixels)")
ax1.set_ylabel("count")

# Create secondary x-axis
# ax2 = ax1.twiny()
# ax2.tick_params(axis='x', labelcolor='r')
# ax2.set_xticks(gt_diagonal_percentile_values)|
# ax2.set_xticklabels(gt_diagonal_percentiles)


# %%
# Prepare data

csv_file = Path(
    "/home/sminano/swc/project_ethology/figs_subset_annotations/run_slurm_1103832_0_17_val_set_full.csv"
)

# read csv
df = pd.read_csv(csv_file)

# add column for percentile
if "full" in csv_file.stem:
    df["percentile"] = [
        map_run_slurm_977884_jobID_to_percentile[run_name.split("_")[-1]]
        for run_name in df["trained_model/run_name"]
    ]
    df["percentile"] = df["percentile"].astype(int)

else:
    df["percentile"] = [
        Path(file).stem.split("_")[-1]
        for file in df["dataset/annotation_files"]
    ]
    df.loc[df["percentile"] == "gen", "percentile"] = "00"
    df["percentile"] = df["percentile"].astype(int)

# check if val or test set
eval_set = "test" if df["cli_args/use_test_set"].all() else "val"
print(f"Evaluating on {eval_set} set")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot precision and recall
fig, ax = plt.subplots(figsize=(10, 6))

# Precision plot
ax.scatter(
    df["percentile"],
    df[f"{eval_set}_precision"],
    marker="o",
    color="blue",
    alpha=0.3,
)
half_width = 0.75
ax.hlines(
    df.groupby("percentile")[f"{eval_set}_precision"].mean().values,
    df.groupby("percentile")[f"{eval_set}_precision"].mean().index
    - half_width,
    df.groupby("percentile")[f"{eval_set}_precision"].mean().index
    + half_width,
    linewidth=4,
    color="blue",
)
ax.set_ylim(0.4, 1.00)
ax.set_xlabel("model trained on bboxes > percentile")
ax.set_ylabel(f"{eval_set} precision", color="blue")
ax.tick_params(axis="y", labelcolor="blue")


# Recall plot
ax2 = ax.twinx()
ax2.scatter(
    df["percentile"],
    df[f"{eval_set}_recall"],
    marker="o",
    color="red",
    alpha=0.3,
)
half_width = 0.75
ax.hlines(
    df.groupby("percentile")[f"{eval_set}_recall"].mean().values,
    df.groupby("percentile")[f"{eval_set}_recall"].mean().index - half_width,
    df.groupby("percentile")[f"{eval_set}_recall"].mean().index + half_width,
    linewidth=4,
    color="red",
)
ax2.set_ylim(0.4, 1.00)
ax2.set_ylabel(f"{eval_set} recall", color="red")
ax2.tick_params(axis="y", labelcolor="red")


plt.show()


# %%
# %matplotlib widget
# %%
