"""Split ``ethology`` annotations using approximate subset sum
================================================================

Splits an annotations dataset based on a grouping variable
(e.g. video) such that no value of the grouping variable appears in both
subsets.
"""

# %%
# Imports
# -------
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pooch
import xarray as xr

from ethology.detectors.datasets import (
    annotations_dataset_to_torch_dataset,
    split_annotations_dataset_group_by,
    split_annotations_dataset_random,
)
from ethology.io.annotations import load_bboxes

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget


# %%
# This notebook demonstrates how to split an annotation dataset into two
# subsets (e.g., train/test) while ensuring that specific categories
# (like videos or species) are kept entirely separate between the subsets.

# The function to do this uses an approximate subset sum approach to find
# which groups (e.g., which videos) to assign to each subset to approximate
# your desired split ratio (like 80/20 or 70/30), while respecting the
# constraint that groups can't be divided.

# This is useful when you want to define a held-out test dataset based on a
# user-specified percentage (e.g., "hold out 20% of samples"). For example,
# you may want to ensure that frames from the same video are not split between
# the training and held-out sets. Using this function ensures complete
# separation of the grouping variable (e.g. video) between the training and
# held-out sets.

# This is in contrast to random splitting, which gives precise proportions,
# but can mix values of the grouping variable across the subsets.

# %%
# Download dataset
# ----------------
# Australian camera trap dataset

data_source = {
    "url": "https://figshare.com/ndownloader/files/53674187",
    "hash": "4019bb11cd360d66d13d9309928195638adf83e95ddec7b0b23e693ec8c7c26b",
}

# Define cache directory
ethology_cache = Path.home() / ".ethology"
ethology_cache.mkdir(exist_ok=True)

# Download the dataset to the cache directory
extracted_files = pooch.retrieve(
    url=data_source["url"],
    known_hash=data_source["hash"],
    fname="ACTD_COCO_files.zip",
    path=ethology_cache,
    processor=pooch.Unzip(extract_dir=ethology_cache / "ACTD_COCO_files"),
)


# %%
# Read as a single annotation dataset
# ------------------------------------

ds_all = load_bboxes.from_files(extracted_files, format="COCO")

print(ds_all)

# %%
# Inspect dataset
# ---------------

# Categories
print(ds_all.map_category_to_str.values())

# Different image sizes
print(np.unique(ds_all.image_shape.values, axis=0))

# Print a few image filenames
print(list(ds_all.map_image_id_to_filename.values())[0])
print(list(ds_all.map_image_id_to_filename.values())[20000])
print(list(ds_all.map_image_id_to_filename.values())[-1])

# Annotation files
print(*ds_all.annotation_files, sep="\n")

# %%
# Split by input file
# -------------------
# Steps:
# - compute array with input annotation file per image
# - run split


# helper function
def split_at_any_delimiter(text: str, delimiters: list[str]) -> list[str]:
    """Split a string at any of the specified delimiters if present."""
    for delimiter in delimiters:
        if delimiter in text:
            return text.split(delimiter)
    return [text]


# Get video-pair filename per image
annotation_file_per_image_id = np.array(
    [
        split_at_any_delimiter(
            ds_all.map_image_id_to_filename[i],
            ["\\"],
        )[0]
        for i in ds_all.image_id.values
    ]
)
print(annotation_file_per_image_id.shape)


# Add to dataset
ds_all["json_file"] = xr.DataArray(
    annotation_file_per_image_id,
    dims="image_id",
)


# %%
# Compute splits without mixing annotation files
# ----------------------------------------------
# We may want to compute a dataset split that does not mix annotation files.
# For example, if we want to split the dataset into train and test sets,
# we may want to ensure that the train set contains only one annotation file
# and the test set contains only one annotation file.

# With ``split_annotations_dataset_group_by`` the dataset is split in two.
# It returns the smallest split first.


fraction_1 = 0.2  # to match output _1, it should be the smaller fraction
fraction_2 = 1 - fraction_1

ds_annotations_1, ds_annotations_2 = split_annotations_dataset_group_by(
    ds_all,
    group_by_var="json_file",
    list_fractions=[fraction_1, fraction_2],
    epsilon=0.01,
)

print(len(ds_annotations_1.image_id.values) / len(ds_all.image_id.values))
print(len(ds_annotations_2.image_id.values) / len(ds_all.image_id.values))

print("--------------------------------")
print(np.unique(ds_annotations_1.json_file.values))
print(np.unique(ds_annotations_2.json_file.values))

# %%
# Another split is 40-60, but there is not that many options with three
# possible json files. Changing the seed doesn't change the split.

fraction_1 = 0.4  # to match output _1, it should be the smaller fraction
fraction_2 = 1 - fraction_1

ds_annotations_1, ds_annotations_2 = split_annotations_dataset_group_by(
    ds_all,
    group_by_var="json_file",
    list_fractions=[fraction_1, fraction_2],
    epsilon=0,
)

print(len(ds_annotations_1.image_id.values) / len(ds_all.image_id.values))
print(len(ds_annotations_2.image_id.values) / len(ds_all.image_id.values))

# %%
# Compute species array
# ----------------------

# Another option is to split by species.
# For this we first need to compute the species array.
species_per_image_id = np.array(
    [
        ds_all.map_image_id_to_filename[i].split("\\")[-2]
        for i in ds_all.image_id.values
    ]
)

# Add the species array to the dataset
ds_all["specie"] = xr.DataArray(
    species_per_image_id,
    dims="image_id",
)


print(*np.unique(species_per_image_id), sep="\n")
print("--------------------------------")
print(f"Total species: {len(np.unique(species_per_image_id))}")


# %%
# Plot frame count per species
# -----------------------------
count_per_specie = dict(Counter(ds_all["specie"].values).most_common())


fig, ax = plt.subplots()
ax.bar(
    count_per_specie.keys(),
    count_per_specie.values(),
)
ax.set_xticks(range(len(count_per_specie)))
ax.set_xticklabels(count_per_specie.keys(), rotation=90)
ax.set_ylabel("# images")
ax.set_title("Image count per specie")


# %%
# Split by species
# ----------------
# Compute an approximate split by species, best of 10

fraction_1 = 0.27
fraction_2 = 1 - fraction_1

ds_species_1, ds_species_2 = split_annotations_dataset_group_by(
    ds_all,
    group_by_var="specie",
    list_fractions=[fraction_1, fraction_2],
    epsilon=0.0,
)

print(len(ds_species_1.image_id.values) / len(ds_all.image_id.values))
print(len(ds_species_2.image_id.values) / len(ds_all.image_id.values))


print("--------------------------------")
print(
    fraction_1
    - len(ds_species_1.image_id.values) / len(ds_all.image_id.values)
)
print(
    fraction_2
    - len(ds_species_2.image_id.values) / len(ds_all.image_id.values)
)

print("--------------------------------")
print(np.unique(ds_species_1.specie.values))
print(np.unique(ds_species_2.specie.values))


# %%
# Compute an approximate split by species, with shuffling

# for seed in [42, 43, 44]:
#     fraction_1 = 0.25
#     fraction_2 = 1 - fraction_1

#     ds_species_1, ds_species_2 = split_annotations_dataset_group_by(
#         ds_all,
#         group_by_var="specie",
#         list_fractions=[fraction_1, fraction_2],
#         epsilon=0.0,
#         seed=seed,
#     )

#     print(f"Seed: {seed}")
#     print(len(ds_species_1.image_id.values) / len(ds_all.image_id.values))
#     print(len(ds_species_2.image_id.values) / len(ds_all.image_id.values))

#     print(np.unique(ds_species_1.specie.values))
#     print(np.unique(ds_species_2.specie.values))

#     print("--------------------------------")


# %%
# Split using random sampling
# ----------------------------
# Alternatively if we want precise splits, we can use random sampling.
# But then we do not have guarantees on the distribution of the splits.
# In this case the default seed is 42.

fraction_1 = 0.27
fraction_2 = 1 - fraction_1

ds_species_1, ds_species_2 = split_annotations_dataset_random(
    ds_all,
    list_fractions=[fraction_1, fraction_2],
    seed=42,
)

print(len(ds_species_1.image_id.values) / len(ds_all.image_id.values))
print(len(ds_species_2.image_id.values) / len(ds_all.image_id.values))
print("--------------------------------")
print(
    fraction_1
    - len(ds_species_1.image_id.values) / len(ds_all.image_id.values)
)
print(
    fraction_2
    - len(ds_species_2.image_id.values) / len(ds_all.image_id.values)
)

# %%
# Unlike random splitting, the approximate subset sum split ensures that
# no value of the grouping category appears in both subsets. For example:
# When grouping by video: each video appears in only the training set or the
# test set, never both
# When grouping by species: each species appears exclusively in one subset

# %%
# Split along another coordinate
# ------------------------------

# For example, split based on bounding box area.
# First compute bounding box area as 1-dimensional array,
# for each annotation id. Then split by annotation id.

# bbox_width_in_pixels =
#   ds_all.shape.sel(space="x")*ds_all.image_shape.sel(space="x")
# bbox_height_in_pixels =
#   ds_all.shape.sel(space="y")*ds_all.image_shape.sel(space="y")

# ds_all["box_area"] = xr.DataArray(
#     # data=(bbox_width_in_pixels*bbox_height_in_pixels),
#     data=ds_all.shape.sel(space="x")*ds_all.shape.sel(space="y"),
#     # dims=('image_id', 'id'),
# )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Convert to torch datasets to train detectors on different subsets

torch_dataset_1 = annotations_dataset_to_torch_dataset(
    ds_species_1, images_directory=extracted_files
)
torch_dataset_2 = annotations_dataset_to_torch_dataset(
    ds_species_2, images_directory=extracted_files
)

# %%
# %%
