"""Convert ``ethology`` annotations to a torch dataset
========================================================

Load bounding box annotations as an ``ethology`` dataset, select a subset of
categories and convert to a
`torch COCO dataset <https://docs.pytorch.org/vision/main/generated/torchvision.datasets.CocoDetection.html>`_.
"""


# %%
# Imports
# -------

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pooch
import torch
import torchvision.transforms.v2.functional as F
from torchvision.utils import draw_bounding_boxes

from ethology.detectors.datasets import annotations_dataset_to_torch_dataset
from ethology.io.annotations import load_bboxes

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget


# %%
# Download dataset
# ------------------
#
# For this example, we will use the dataset from the
# `UAS Imagery of Migratory Waterfowl at New Mexico Wildlife Refuges <https://lila.science/datasets/uas-imagery-of-migratory-waterfowl-at-new-mexico-wildlife-refuges/>`_.
# This dataset is part of the `Drones For Ducks project
# <https://aspire.unm.edu/research/funded-research/ducks-and-drones.html>`_
# that aims to develop an efficient method to count and identify species of
# migratory waterfowl at wildlife refuges across New Mexico.
#
# The dataset is made up of a set of drone images and corresponding
# bounding box annotations. Annotations are provided by both expert
# annotators and volunteers.
#
# Since the dataset is not very large, we can download it as a zip file
# directly from the URL provided in the dataset webpage.
# We use the `pooch <https://github.com/fatiando/pooch/>`_ library
# to download it to the ``.ethology`` cache directory.


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


# %%
# For this example, we will focus on the annotations labelled by the experts.

data_dir = ethology_cache / "uas-imagery-of-migratory-waterfowl"
experts_dir = data_dir / "experts"

annotations_file = experts_dir / "20230331_dronesforducks_expert_refined.json"
images_dir = experts_dir / "images"


# %%
# Load annotations as `ethology` dataset
# --------------------------------------

ds = load_bboxes.from_files(
    annotations_file, images_dirs=images_dir, format="COCO"
)

print(ds)
print(ds.sizes)

# %%
# Transform image filenames
# -------------------------

# Image filenames in input file are .JPG but files are .jpg
# Change image filenames dict to .jpg
map_image_id_to_filename = {
    k: v.replace(".JPG", ".jpg")
    for k, v in ds.map_image_id_to_filename.items()
}
ds.attrs["map_image_id_to_filename"] = map_image_id_to_filename


# %%
# Count annotations per category
# -------------------------------

list_category_counts = [
    (ky, val, (ds.category == ky).sum().item())
    for ky, val in ds.map_category_to_str.items()
]

# Sort by decreasing count
list_category_counts.sort(key=lambda x: x[2], reverse=True)


# %%
# Select a subset dataset
# -----------------------
# Make a new dataset with only the bottom/top 3 categories

# Compute the categories to keep
n_categories = 2
categories_to_keep = [x[0] for x in list_category_counts[:n_categories]]
print(f"Categories to keep: {categories_to_keep}")

# Compute categories mask array
# True where categories are in the set to keep, False otherwise
categories_mask = ds.category.isin(categories_to_keep)  # dim: image_id, id

ds_subset = ds.where(categories_mask, drop=True)

# inspect
print(f"ds_subset unique categories: {np.unique(ds_subset.category.values)}")
print(f"ds_subset.sizes: {ds_subset.sizes}")  # note reduced dimensions
print(f"ds_subset.image_shape shape: {ds_subset.image_shape.shape}")

# %%
# Note that due to the underlying broadcasting in the ``where`` operation,
# the image_shape array now has ``image_id``, ``space``, and ``id`` dimensions
# and the ``category`` array is now a float. For clarity we go back to the
# ``ethology``
# convention and make the ``category`` array and integer one, with -1 for empty
# values, and the ``image_shape`` array an integer one with only ``image_id``
# and ``space`` dimensions.

ds_subset["category"] = ds_subset.category.fillna(-1).astype(int)
ds_subset["image_shape"] = ds.image_shape
# this assignment takes only the (image_id, space) coordinates from
# ds.image_shape that are also present in ds_subset


print(ds_subset)
print("---------")
print(f"ds_subset.image_shape shape: {ds_subset.image_shape.shape}")


# %%
# Convert dataset to torch dataset
# -----------------------------------

dataset_torch = annotations_dataset_to_torch_dataset(ds_subset)


# %%
# Sample from the torch dataset and convert bbox format
# -----------------------------------------------------

# get one image and its annotations
sample_idx = 2
img, annot = dataset_torch[sample_idx]

# annot is a list of bboxes dictionaries
# coords are in XYWH format
bboxes_tensor_xywh = torch.as_tensor([ann["bbox"] for ann in annot])
print(f"Bbox in XYWH format: {bboxes_tensor_xywh[0, :]}")

# convert bbox format from XYWH to XYXY
bboxes_tensor_xyxy = F.convert_bounding_box_format(
    bboxes_tensor_xywh,
    old_format="XYWH",
    new_format="XYXY",
)
print(f"Bbox in XYXY format: {bboxes_tensor_xyxy[0, :]}")


# %%
# Visualize selected sample using torchvision ``draw_bounding_boxes``
# --------------------------------------------------------------------
# From https://docs.pytorch.org/vision/0.21/auto_examples/others/plot_visualization_utils.html


# map category ID to color
cmap = plt.cm.tab10
map_category_id_to_color_ints = {
    i: tuple((np.array(cmap(i)[:3]) * 255).astype(int))
    for i in np.unique(ds_subset.category.values)
    if i != -1
}
map_category_id_to_color_floats = {
    i: tuple(np.array(cmap(i)[:3]))
    for i in np.unique(ds_subset.category.values)
    if i != -1
}

# list of categories per annotation in image
list_category_ids_in_image = [ann["category_id"] for ann in annot]

# create image with boxes
img_with_boxes = draw_bounding_boxes(
    F.pil_to_tensor(img),
    bboxes_tensor_xyxy,
    colors=[
        map_category_id_to_color_ints[x] for x in list_category_ids_in_image
    ],
    width=15,
)

# plot
fig, ax = plt.subplots()
ax.imshow(img_with_boxes.permute(1, 2, 0))
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")


# add legend
legend_elements = [
    plt.Line2D([0], [0], color=c)
    for c in map_category_id_to_color_floats.values()
]
legend_labels = [
    ds_subset.map_category_to_str[x]
    for x in np.unique(ds_subset.category.values)
    if x != -1
]
plt.legend(
    legend_elements,
    legend_labels,
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)
plt.tight_layout()
