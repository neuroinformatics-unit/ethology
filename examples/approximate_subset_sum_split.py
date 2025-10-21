"""Split an annotations dataset by category
==============================================

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

from ethology.datasets.split import (
    split_dataset_group_by,
    split_dataset_random,
)
from ethology.io.annotations import load_bboxes

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget


# %%
# This notebook demonstrates how to split an annotation dataset into two
# subsets (e.g., train/test) while ensuring that specific categories
# (like videos or species) are kept entirely separate between the subsets.
#
# This is useful when you want to define a held-out test dataset based on a
# user-specified percentage (e.g., "hold out 20% of samples"). For example,
# you may want to ensure that frames from the same video are not split between
# the training and held-out sets. Using this function ensures complete
# separation of the grouping variable (e.g. video) between the training and
# held-out sets.
#
# This is in contrast to random splitting, which gives precise proportions,
# but can mix values of the grouping variable across the subsets. Both
# approaches are useful in different situations, and this notebook shows how
# to apply them using ``ethology`` utilities.

# %%
# Download dataset
# ----------------
# For this example, we will use the `Australian Camera Trap Dataset
# <https://figshare.com/articles/dataset/Australian_Camera_Trap_Data_ACTD_/27177912>`_
# which comprises images from camera traps across various sites in Victoria,
# Australia.
#
# We use the `pooch <https://github.com/fatiando/pooch/>`_ library to download
# the dataset to the ``.ethology`` cache directory.

# %%
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

print(*extracted_files, sep="\n")

# %%
# Read as a single annotation dataset
# ------------------------------------
#
# The dataset contains three different COCO annotation files. We can load them
# as a single dataset using the
# :func:`ethology.io.annotations.load_bboxes.from_files`
# function.


# %%
ds_all = load_bboxes.from_files(extracted_files, format="COCO")

print(ds_all)
print(*ds_all.annotation_files, sep="\n")


# %%
# Inspect dataset
# ---------------
#
# The combined dataset contains annotations for 39426 images,
# with each image having a maximum of 6 annotations. We can further inspect
# the different categories considered, the image sizes and the format of the
# image filenames.

# %%

# Categories
print("Categories:")
print(ds_all.map_category_to_str.values())
print("--------------------------------")

# Image sizes
print("Image sizes:")
print(np.unique(ds_all.image_shape.values, axis=0))
print("--------------------------------")

# Print a few image filenames
print("Sample image filenames:")
print(list(ds_all.map_image_id_to_filename.values())[0])
print(list(ds_all.map_image_id_to_filename.values())[30000])
print(list(ds_all.map_image_id_to_filename.values())[-1])

# %%
# The image filenames encode a bit more of extra information, such
# as the original annotation file or the species class. We can use
# this to define possible grouping variables for the images in the dataset.


# %%
# Split by input annotation file
# -------------------------------
# Let's assume we want to split the dataset into two sets,
# such that each set has distinct annotation files. This may be useful for
# example, if we want to split the dataset into train and test sets, while
# ensuring that the test set contains only annotations from one file.
#
# To do this, we first need to compute the annotation file per image.
# Then we can split the images in the dataset based on the annotation file.
#
# We will use a helper function to extract the information of interest
# from the image filenames.

# %%


# Helper function
def split_at_any_delimiter(text: str, delimiters: list[str]) -> list[str]:
    """Split a string at any of the specified delimiters if present."""
    for delimiter in delimiters:
        if delimiter in text:
            return text.split(delimiter)
    return [text]


# Get annotation file per image
annotation_file_per_image_id = np.array(
    [
        split_at_any_delimiter(
            ds_all.map_image_id_to_filename[i],
            ["\\"],
        )[0]
        for i in ds_all.image_id.values
    ]
)

# Add to dataset
ds_all["json_file"] = xr.DataArray(
    annotation_file_per_image_id, dims="image_id"
)


# %%
# Now that we have the 1-dimensional array with the annotation file per image
# along the ``image_id`` dimension, we can split the dataset using the
# :func:`ethology.detectors.datasets.split_annotations_dataset_group_by`
# function, which implements an
# `approximate subset sum algorithm
# <https://en.wikipedia.org/wiki/Subset_sum_problem#Fully-polynomial_time_approximation_scheme>`_.
#
# The function accepts an optional ``epsilon`` parameter. This is the
# percentage of the optimal solution that the solution is guaranteed to be
# within. If ``epsilon`` is 0, the solution will be the best solution
# (optimal) within the constraints.
#
# The subsets are returned in the same order as the input list of
# fractions. The algorithm computes the smallest subset to be less than
# or equal to the requested fraction.

# %%
fraction_1 = 0.2
fraction_2 = 1 - fraction_1

ds_annotations_1, ds_annotations_2 = split_dataset_group_by(
    ds_all,
    group_by_var="json_file",
    list_fractions=[fraction_1, fraction_2],
    epsilon=0,
)

# %%
# We can verify that the size of the subsets obtainedis close to the requested
# fractions. Since we used the default ``epsilon=0``, this split is
# the best solution we can get within the specified constraints.

# %%
print(f"User specified fractions:{[fraction_1, fraction_2]}")

print("Split fractions:")
print(len(ds_annotations_1.image_id.values) / len(ds_all.image_id.values))
print(len(ds_annotations_2.image_id.values) / len(ds_all.image_id.values))

# %%
# We can also verify that the subsets contain distinct annotation files.

# %%
print(f"Subset 1 files: {np.unique(ds_annotations_1.json_file.values)}")
print(f"Subset 2 files: {np.unique(ds_annotations_2.json_file.values)}")

# %%
# In this case there are only three possible splits of the dataset, since
# there are only three possible values for the source annotation file.
#
# In more complex cases with many inputs and many possible splits, the
# computation may be slow. In this case, we may want to use a
# larger ``epsilon`` value, to get faster to a solution that is close
# enough to the optimal one. The choice of epsilon involves a trade-off
# between accuracy and speed.


# %%
# Split by species
# ----------------------
# Let's consider another case, in which we would like to split the images in
# the dataset by species.
#
# As before, we first compute the species array for each image, which we
# derive from the image filename. Then we add the species array to the dataset.

# %%

# Get species name per image
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

print(f"Total species: {len(np.unique(species_per_image_id))}")


# %%
# We have 15 different species in the dataset. With a bar plot we can visualise
# their distribution in the dataset.

# %%
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
plt.tight_layout()


# %%
# We can now split the dataset by species using the
# :func:`ethology.detectors.datasets.split_annotations_dataset_group_by`
# function again. For example, for a 27/73 split, we would do:

# %%
fraction_1 = 0.27
fraction_2 = 1 - fraction_1

ds_species_1, ds_species_2 = split_dataset_group_by(
    ds_all,
    group_by_var="specie",
    list_fractions=[fraction_1, fraction_2],
    epsilon=0,
)

# %%
# We can check how close is the resulting split to the requested fractions,
# and verify that the subsets contain distinct species.

# %%
print(f"User specified fractions:{[fraction_1, fraction_2]}")

print("Split fractions:")
print(len(ds_species_1.image_id.values) / len(ds_all.image_id.values))
print(len(ds_species_2.image_id.values) / len(ds_all.image_id.values))


print("Difference in fraction_1:")
print(
    abs(
        fraction_1
        - len(ds_species_1.image_id.values) / len(ds_all.image_id.values)
    )
)

print("--------------------------------")
print(f"Subset 1 species: {np.unique(ds_species_1.specie.values)}")
print(f"Subset 2 species: {np.unique(ds_species_2.specie.values)}")


# %%
# Split using random sampling
# ----------------------------
# Very often we want to compute splits for a specific fraction, and
# don't care if a grouping variables (e.g. species or source video) are mixed
# across subsets. This may be the case for example when splitting for
# validation and training sets. In this case, we can use random sampling with
# the function
# :func:`ethology.detectors.datasets.split_annotations_dataset_random`.
# By setting a different value for the random ``seed``, we can get
# different splits with the same requested fractions.

# %%
fraction_1 = 0.27
fraction_2 = 1 - fraction_1

ds_species_1, ds_species_2 = split_dataset_random(
    ds_all,
    list_fractions=[fraction_1, fraction_2],
    seed=42,
)


print(f"User specified fractions:{[fraction_1, fraction_2]}")

print("Split fractions:")
print(len(ds_species_1.image_id.values) / len(ds_all.image_id.values))
print(len(ds_species_2.image_id.values) / len(ds_all.image_id.values))
