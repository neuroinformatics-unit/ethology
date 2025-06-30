"""A script to regenerate a labelled dataset from a VIA JSON file."""

# %%
# Imports
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from movement.plots import plot_occupancy
from movement.roi import PolygonOfInterest

from ethology.annotations.io import load_bboxes, save_bboxes

# %%
# %matplotlib widget
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data

via_path = Path(
    "/home/sminano/swc/project_ethology/sept2023_annotations.bk/VIA_JSON_combined.json"
)

# coco_path =

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load data

ds = load_bboxes.from_files(via_path, format="VIA")

ds_as_movement = ds.rename_dims({"image_id": "time", "id": "individuals"})

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Count total number of annotations

total_annotations_w_nans = ds.position.shape[0] * ds.position.shape[2]
total_annotations = (
    total_annotations_w_nans
    - ds.position.isnull().any(dim=["space"]).sum().item()
)

# print(ds.position.isnull().any(dim=["space"]).shape)
print(total_annotations_w_nans)
print(total_annotations)  # should be 53041

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot labels occupancy
fig, ax_occupancy, hist = plot_occupancy(
    ds_as_movement.position, bins=[100, 50]
)
ax_occupancy.set_xlabel("x (pixels)")
ax_occupancy.set_ylabel("y (pixels)")
ax_occupancy.axis("equal")
ax_occupancy.invert_yaxis()

# # get the colorbar
# cbar = fig.colorbar(hist["counts"], ax=ax_occupancy)
# cbar.set_label("count")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot spatial distribution of individuals
fig, ax_spatial = plt.subplots()
ax_spatial.scatter(
    ds.position.sel(space="x").values,
    ds.position.sel(space="y").values,
    s=1,
    alpha=0.5,
)
ax_spatial.set_xlabel("x (pixels)")
ax_spatial.set_ylabel("y (pixels)")
ax_spatial.axis("equal")
ax_spatial.invert_yaxis()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Count labels within a polygon

# Define polygon
ymin, ymax = 0, 1500
xmin, xmax = 0, 4100
central_region = PolygonOfInterest(
    ((xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin)),
    name="Central region",
)

# plot on spatial distribution map
central_region.plot(ax_spatial, facecolor="red", alpha=0.25)

# OJO ds.position contains nans -- they are not in the polygon
# print(ds.position.isnull().sum())

# check labels in the polygon
ds_in_region = central_region.contains_point(ds.position)
print(ds_in_region.sum())
print(ds_in_region.sum() / total_annotations)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute a histogram of bbox size

bbox_diagonal = np.sqrt(
    ds.shape.sel(space="x") ** 2 + ds.shape.sel(space="y") ** 2
)
slc_nan_diagonal = bbox_diagonal.isnull()

fig, ax_hist = plt.subplots()


# ignore nans
ax_hist.hist(
    bbox_diagonal.values.flatten()[~slc_nan_diagonal.values.flatten()],
    bins=100,
)
ax_hist.set_xlabel("bbox diagonal (pixels)")
ax_hist.set_ylabel("count")

# add a vertical line at the median
ax_hist.axvline(np.nanmedian(bbox_diagonal), color="r", linestyle="--")
ax_hist.axvline(np.nanmean(bbox_diagonal), color="k", linestyle="-")

# add legend
ax_hist.legend(["median", "mean"], loc="upper right")

print(np.nanmedian(bbox_diagonal))
print(np.nanmean(bbox_diagonal))

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot spatial distribution of individuals and color by bbox diagonal
fig, ax_spatial = plt.subplots()
sc = ax_spatial.scatter(
    ds.position.sel(space="x").values,
    ds.position.sel(space="y").values,
    s=1,
    alpha=0.5,
    c=bbox_diagonal.values,
    cmap="viridis",
)
ax_spatial.set_xlabel("x (pixels)")
ax_spatial.set_ylabel("y (pixels)")
ax_spatial.axis("equal")
ax_spatial.invert_yaxis()

# add colorbar
cbar = ax_spatial.figure.colorbar(sc)
cbar.set_label("bbox diagonal (pixels)")
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Compute percentiles of bbox diagonal
bbox_diagonal_percentiles = {}
for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]:
    bbox_diagonal_percentiles[p] = np.percentile(
        bbox_diagonal.values.flatten()[~slc_nan_diagonal.values.flatten()],
        p,
    )

print(bbox_diagonal_percentiles)


# %%%%%%%%%%%%%%
# Plot annotations with bbox diagonal below a certain percentile
selected_percentile = 1

slc_flattened = (
    bbox_diagonal <= bbox_diagonal_percentiles[selected_percentile]
).values.flatten()

print(slc_flattened.sum())
print(slc_flattened.sum() / total_annotations)
print(bbox_diagonal_percentiles[selected_percentile])

fig, ax = plt.subplots()
sc = ax.scatter(
    ds.position.sel(space="x").values,
    ds.position.sel(space="y").values,
    s=1,
    alpha=0.5,
    c=bbox_diagonal.values,
    cmap="viridis",
)
ax.scatter(
    ds.position.sel(space="x").values.flatten()[slc_flattened],
    ds.position.sel(space="y").values.flatten()[slc_flattened],
    s=10,
    c="r",
)
ax.set_title(f"Percentile {selected_percentile}")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.axis("equal")
ax.invert_yaxis()
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export only small annotations and visualize in VIA tool
# TODO: adapt to xarray dataset
ds["diagonal"] = np.sqrt(
    ds.shape.sel(space="x") ** 2 + ds.shape.sel(space="y") ** 2
)

for p in [1, 5, 10, 25, 50]:
    ds_small = ds.loc[ds.diagonal <= bbox_diagonal_percentiles[p]]

    # Export as VIA tracks file
    out = save_bboxes.to_COCO_file(
        ds_small,
        Path(f"annotations_below_percentile_{p}.json"),
    )

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Export all but small annotations
for p in [1, 5, 10, 25, 50]:
    ds_large = ds.loc[ds.diagonal > bbox_diagonal_percentiles[p]]

    # Set category_id to 1 -- 0 is reserved for background
    ds_large.loc[:, "category_id"] = 1

    # Export as COCO annotations file
    out = save_bboxes.to_COCO_file(
        ds_large,
        Path(f"annotations_above_percentile_{p}.json"),
    )

# %%
