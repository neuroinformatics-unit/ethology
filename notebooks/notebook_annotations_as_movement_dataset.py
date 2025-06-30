"""Example notebook showing how to use `movement` utilities with annotation datasets."""

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from movement.io import save_poses
from movement.plots import plot_occupancy
from movement.roi import PolygonOfInterest

from ethology.annotations.io import load_bboxes

# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Multiple files
ds = load_bboxes.from_files(
    [
        (
            Path.home() / ".ethology-test-data/test_annotations/"
            "medium_bboxes_dataset_VIA/VIA_JSON_sample_1.json"
        ),
        (
            Path.home() / ".ethology-test-data/test_annotations/"
            "medium_bboxes_dataset_VIA/VIA_JSON_sample_2.json"
        ),
    ],
    format="VIA",
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Single file
ds = load_bboxes.from_files(
    (
        Path.home() / ".ethology-test-data/test_annotations/"
        "medium_bboxes_dataset_VIA/VIA_JSON_sample_2.json"
    ),
    format="VIA",
)


# %%%%%%%%%%%%%%%%%%%%%%%%%
# Rename dimensions to match movement dataset

ds_as_movement = ds.rename_dims({"image_id": "time", "id": "individuals"})

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot labels occupancy
fig, ax, hist = plot_occupancy(ds_as_movement.position, bins=[100, 50])
ax.set_label("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.axis("equal")
ax.invert_yaxis()

# # add colorbar
# cbar = plt.colorbar()
# cbar.set_label('Occupancy')
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot labels

# plot_centroid_trajectory(ds_as_movement.position) ---odd

fig, ax = plt.subplots()
ax.scatter(
    ds.position.sel(space="x").values,
    ds.position.sel(space="y").values,
    s=1,
    alpha=0.5,
)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.axis("equal")
ax.invert_yaxis()

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# plot bboxes as ROIs?

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# labels within a polygon
central_region = PolygonOfInterest(
    ((1000, 500), (1000, 1250), (2500, 1250), (2500, 500)),
    #  ((2120, 0),(2120, 80), (2220, 80), (2220, 0)),
    # there should be 4 inside here ^
    name="Central region",
)

# plot on occupancy map
central_region.plot(ax, facecolor="red", alpha=0.25)

# OJO ds.position contains nans -- they are not in the polygon
print(ds.position.isnull().sum())

# check labels in the polygon
ds_in_region = central_region.contains_point(ds.position)
print(ds_in_region.sum())

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# For kpts: plot in egocentric coord syst


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export as movement poses dataset
# (this allows us to visualise the dataset with movement-napari utilities)

# as a pose dataset


# copy ds_as_movement
ds_as_movement_to_export = ds_as_movement.copy()

# drop arrays that are not needed for the pose dataset
ds_as_movement_to_export = ds_as_movement_to_export.drop_vars(
    ["shape", "category", "category_id"]
)

# Add a new dimension (keypoints)
ds_as_movement_to_export = ds_as_movement_to_export.expand_dims(
    dim={"keypoints": ["centroid"]}, axis=2
)


# add confidence array
ds_as_movement_to_export["confidence"] = xr.DataArray(
    np.full(
        (
            ds_as_movement_to_export.sizes["time"],
            ds_as_movement_to_export.sizes["keypoints"],
            ds_as_movement_to_export.sizes["individuals"],
        ),
        np.nan,
    ),
    dims=["time", "keypoints", "individuals"],
)


# save as sleap analysis file
save_poses.to_sleap_analysis_file(ds_as_movement_to_export, "test_poses.h5")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export as movement bboxes dataset

# copy ds_as_movement
ds_as_movement_to_export = ds_as_movement.copy()

# drop arrays that are not needed for the bboxes dataset
ds_as_movement_to_export = ds_as_movement_to_export.drop_vars(
    ["category", "category_id"]
)


# add confidence array
ds_as_movement_to_export["confidence"] = xr.DataArray(
    np.full(
        (
            ds_as_movement_to_export.sizes["time"],
            ds_as_movement_to_export.sizes["individuals"],
        ),
        np.nan,
    ),
    dims=["time", "individuals"],
)

# save as VIA tracks file
# save_poses.to_via_tracks_file(ds_as_movement_to_export, "test_bboxes.h5")

# %%
