"""Load bounding box annotations as an ``ethology`` dataset
========================================================

Load bounding box annotations as an ``ethology`` dataset and visualize it.
"""


# %%
# Imports
# -------

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pooch
import xarray as xr
from movement.io import save_bboxes
from movement.plots import plot_occupancy
from movement.roi import PolygonOfInterest

from ethology.io.annotations import load_bboxes

# %%
# Download dataset
# ------------------

# Source of the dataset
data_source = {
    "url": "https://storage.googleapis.com/public-datasets-lila/uas-imagery-of-migratory-waterfowl/uas-imagery-of-migratory-waterfowl.20240220.zip",
    "hash": "c5b8dfc5a87ef625770ac8f22335dc9eb8a67688b610490a029dae81815a9896",
}

# Define cache directory
ethology_cache = Path.home() / ".ethology"
ethology_cache.mkdir(exist_ok=True)

# Download the dataset to the cache directory
extracted_files = pooch.retrieve(
    url=data_source["url"],
    known_hash=data_source["hash"],
    fname="waterfowl_dataset.zip",
    path=ethology_cache,
    processor=pooch.Unzip(extract_dir=ethology_cache),
)

data_dir = ethology_cache / "uas-imagery-of-migratory-waterfowl"


# %%
# Get annotations file
# -----------------------

annotations_file = (
    data_dir / "experts" / "20230331_dronesforducks_expert_refined.json"
)


# %%
# Load annotations as ``ethology`` dataset
# --------------------------------------

ds = load_bboxes.from_files(annotations_file, format="COCO")

print(ds)
print(ds.sizes)


# %%
# Plot all annotations and color by category
# ------------------------------------------

print(f"Number of categories: {len(ds.map_category_to_str)}")
cmap = plt.cm.tab10

fig, ax = plt.subplots()
sc = ax.scatter(
    ds.position.sel(space="x").values,
    ds.position.sel(space="y").values,
    s=3,
    c=ds.category.values,
    cmap=cmap,
)

# add legend
legend_elements = [
    plt.Line2D([0], [0], color=cmap(i))
    for i in np.unique(ds.category.values)
    # we use ds.category.values rather than
    # ds.map_category_to_str.values() because
    # the latter contains the padding value -1
]
plt.legend(
    legend_elements,
    ds.map_category_to_str.values(),
    bbox_to_anchor=(1, 1),
    loc="upper left",
)

ax.set_title("Annotations per category")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.axis("equal")
ax.invert_yaxis()

# %%
# Compute annotations within a polygon
# -------------------------------------

# define a polygon
central_region = PolygonOfInterest(
    ((1000, 500), (1000, 3000), (4500, 3000), (4500, 500)),
    name="Central region",
)

# plot all annotations
fig, ax = plt.subplots()
sc = ax.scatter(
    ds.position.sel(space="x").values,
    ds.position.sel(space="y").values,
    s=3,
    c=ds.category.values,
    cmap=cmap,
)
ax.set_title("Annotations in polygon")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.axis("equal")
ax.invert_yaxis()


# plot polygon on top
central_region.plot(ax, facecolor="red", edgecolor="red", alpha=0.25)

# Check number of annotations in the polygon
# Note that if position is nan, ``.contains_point`` returns False
ds_in_region = central_region.contains_point(
    ds.position
)  # (n_images, n_max_annotations_per_image)

n_annotations_in_region = ds_in_region.sum()
n_annotations_total = (~ds.position.isnull().any(axis=1)).sum()
fraction_in_region = n_annotations_in_region / n_annotations_total

print(f"Total annotations: {n_annotations_total.item()}")
print(f"Annotations in region: {n_annotations_in_region.item()}")
print(f"Fraction of annotations in region: {fraction_in_region * 100:.2f}%")

# %%
# Transform to ``movement``-like dataset
# ---------------------------------------
# rename dimensions
ds_as_movement = ds.rename({"image_id": "time", "id": "individuals"})

# rename 'individuals' coordinate values
ds_as_movement["individuals"] = [
    f"id_{i.item()}" for i in ds_as_movement.individuals.values
]

# add confidence array
ds_as_movement["confidence"] = xr.DataArray(
    np.full(
        (
            ds_as_movement.sizes["time"],
            ds_as_movement.sizes["individuals"],
        ),
        np.nan,
    ),
    dims=["time", "individuals"],
)

# add time_unit
# TODO: add to bboxes validator?
ds_as_movement.attrs["time_unit"] = "frames"


# check if valid ValidBboxesDataset
# valid_ds = _validate_dataset(ds_as_movement, ValidBboxesDataset)
# print(valid_ds)

print(ds_as_movement)
print(ds_as_movement.sizes)


# %%
# Plot occupancy map using ``movement`` utilities
# -----------------------------------------------

# We use the image shape to determine the number of bins along each dimension
# to make the occupancy map more informative.
image_width = np.unique(ds["image_shape"].sel(space="x").values).item()
image_height = np.unique(ds["image_shape"].sel(space="y").values).item()

image_AR = image_width / image_height

n_bins_x = 75
n_bins_y = int(n_bins_x / image_AR)

# plot occupancy map
fig, ax, hist = plot_occupancy(
    ds_as_movement.position,
    bins=[n_bins_x, n_bins_y],
)
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
ax.axis("equal")
ax.invert_yaxis()


# %%
# Export as movement bboxes dataset
# ---------------------------------
# This allows us to visualise the dataset in the ``movement`` napari plugin.

save_bboxes.to_via_tracks_file(ds_as_movement, "waterfowl_dataset.csv")


# %%
# Clean-up
# --------

os.remove("waterfowl_dataset.csv")
