"""Convert bounding box annotations to a torch dataset
========================================================

Load bounding box annotations as an `ethology` dataset, modify it and
convert to a torch dataset.
"""


# %%
# Imports
# -------

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2.functional as F
from torchvision.utils import draw_bounding_boxes

from ethology.io.annotations import load_bboxes
from ethology.torch_dataset import from_annotations_dataset

# %%
# Load sample dataset
# -------------------
annotations_file = (
    "/home/sminano/swc/project_ethology/COCO_readable/"
    "uas-imagery-of-migratory-waterfowl/experts/20230331_dronesforducks_expert_refined.json"
)
images_dir = Path(annotations_file).parent / "images"


font_path = Path(torch.__path__[0]) / "qt" / "fonts" / "DejaVuSans.ttf"


# %%
# Helper function to show images
# ------------------------------


def show(imgs: torch.Tensor | list[torch.Tensor]):
    """Plot input tensors as images."""
    if not isinstance(imgs, list):
        imgs = [imgs]
    fig, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img))
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])


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
# Create a subset dataset
# -----------------------

# Make a new dataset with only a subset of categories

list_category_counts = [
    (ky, val, (ds.category == ky).sum().item())
    for ky, val in ds.map_category_to_str.items()
]

# Sort by count
list_category_counts.sort(key=lambda x: x[2], reverse=True)

# Make a new dataset with only the top 3 categories
keys2keep = [x[0] for x in list_category_counts[:3]]
ds_top3 = ds.where(ds.category.isin(keys2keep), drop=True)


print(ds_top3)


# %%
# Convert dataset to torch dataset
# -----------------------------------

dataset_torch = from_annotations_dataset(ds_top3)

# %%
# Visualize dataset
# -----------------------------------

# From https://docs.pytorch.org/vision/0.21/auto_examples/others/plot_visualization_utils.html


# get one image and its annotations
sample_idx = 1
img, annot = dataset_torch[sample_idx]

# transform to torch tensors
img_tensor = F.pil_to_tensor(img)
bboxes_tensor = F.convert_bounding_box_format(
    torch.as_tensor([ann["bbox"] for ann in annot]),
    old_format="XYWH",
    new_format="XYXY",
)

# map category ID to color
cmap = plt.cm.tab10
map_category_id_to_color = {
    i: tuple((np.array(cmap(i)[:3]) * 255).astype(int))
    for i in ds_top3.map_category_to_str
}
list_colors = [map_category_id_to_color[ann["category_id"]] for ann in annot]


# plot
drawn_boxes = draw_bounding_boxes(
    img_tensor,
    bboxes_tensor,
    colors=list_colors,
    width=15,
)
show(drawn_boxes)

# add legend
legend_elements = [
    plt.Line2D([0], [0], color=cmap(i)) for i in ds_top3.map_category_to_str
]
plt.legend(
    legend_elements,
    ds_top3.map_category_to_str.values(),
    bbox_to_anchor=(1.05, 1),
    loc="upper left",
)
plt.tight_layout()

# %%
