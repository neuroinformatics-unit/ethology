"""Convert bounding boxes to masks using SAM2
===============================================

Use `SAM2 <https://github.com/facebookresearch/sam2>`_ to generate instance
segmentation masks from bounding box annotations, save them to a
`zarr <https://zarr.readthedocs.io/>`_ store, and read the masks back aligned
with the original bounding boxes dataset.

.. note::
   This example requires the ``sam2`` package and a CUDA or MPS device for
   running SAM2 inference. If neither is available, pre-computed masks are
   downloaded instead. Install ``sam2`` with ``pip install sam2``.
"""


# %%
# Imports
# -------

import shutil
import tempfile
from pathlib import Path

import dask
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import pooch
import torch
import xarray as xr
import zarr
from PIL import Image

from ethology.io.annotations import load_bboxes

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget


# %%
# Download dataset
# ----------------
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

data_dir = ethology_cache / "uas-imagery-of-migratory-waterfowl"


# %%
# For this example, we will focus on the annotations labelled by the experts.

annotations_file = (
    data_dir / "experts" / "20230331_dronesforducks_expert_refined.json"
)
images_dir = data_dir / "experts" / "images"


# %%
# Load annotations as an ``ethology`` dataset
# --------------------------------------------
#
# We can use the :func:`~ethology.io.annotations.load_bboxes.from_files`
# function to load the COCO file with the
# expert annotations as an ``ethology`` dataset.

ds = load_bboxes.from_files(
    annotations_file, format="COCO", images_dirs=images_dir
)

print(ds)
print(ds.sizes)


# %%
# Load images as a lazy dataset variable
# ----------------------------------------
#
# We load the images as a dask-backed :class:`xarray.DataArray` variable
# in the dataset. This means images are only loaded from disk when accessed,
# keeping memory usage low. The image paths are derived from the dataset's
# ``map_image_id_to_filename`` attribute, ensuring alignment with the
# ``image_id`` dimension.

# Get image paths aligned with image_id
image_paths = [
    images_dir / ds.map_image_id_to_filename[i]
    for i in range(ds.sizes["image_id"])
]

# Sample first image for shape
sample = Image.open(image_paths[0])
img_w, img_h = sample.size
img_c = len(sample.getbands())


def _load_image(path):
    """Load a single image as a numpy array."""
    return np.array(Image.open(path))


# Build dask array of lazily-loaded images
lazy_images = [
    da.from_delayed(
        dask.delayed(_load_image)(path),
        shape=(img_h, img_w, img_c),
        dtype=np.uint8,
    )
    for path in image_paths
]

ds["image"] = xr.DataArray(
    data=da.stack(lazy_images, axis=0),
    dims=["image_id", "img_h", "img_w", "channel"],
)

print(ds)


# %%
# Convert bounding boxes to SAM2 format
# ----------------------------------------
#
# The ``ethology`` dataset stores bounding boxes as centre coordinates
# (``position``) and dimensions (``shape``). SAM2 expects bounding boxes
# in corner format: ``[x1, y1, x2, y2]``, where ``(x1, y1)`` is the
# top-left corner and ``(x2, y2)`` is the bottom-right corner.

# Compute corners from centre + shape
x1y1 = ds.position - ds.shape / 2
x2y2 = ds.position + ds.shape / 2

# Build a dict mapping image_id to (N, 4) arrays of [x1, y1, x2, y2]
# dropping NaN-padded entries
map_image_id_to_boxes = {
    idx: np.c_[
        x1y1.sel(image_id=idx).dropna(dim="id", how="all").values.T,
        x2y2.sel(image_id=idx).dropna(dim="id", how="all").values.T,
    ]
    for idx in range(ds.sizes["image_id"])
}


# %%
# Generate masks with SAM2
# -------------------------
#
# We check if a suitable device (CUDA or MPS) is available for running
# SAM2 inference. If so, we run SAM2 to generate boolean masks for each
# bounding box. Otherwise, we download pre-computed masks.
#
# The masks are stored as a 4D boolean zarr array with shape
# ``(n_images, n_max_ids, img_h, img_w)``, where each slice
# ``[image_id, id]`` is a binary mask for that bounding box.
# Unoccupied id slots (images with fewer annotations than the maximum)
# are filled with ``False``.

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = None

# Create a temporary directory for the zarr store
tmp_dir = Path(tempfile.mkdtemp())
zarr_path = tmp_dir / "masks.zarr"

n_images = ds.sizes["image_id"]
n_max_ids = ds.sizes["id"]

if device is not None:
    from sam2.sam2_image_predictor import SAM2ImagePredictor

    # Load SAM2 predictor
    predictor = SAM2ImagePredictor.from_pretrained(
        "facebook/sam2.1-hiera-base-plus", device=device
    )

    # Create zarr store for boolean masks
    mask_zarr = zarr.open(
        str(zarr_path),
        mode="w",
        shape=(n_images, n_max_ids, img_h, img_w),
        dtype="bool",
        fill_value=False,
        chunks=(1, n_max_ids, img_h, img_w),
    )

    # Process images in batches
    batch_size = 1
    for idx in range(0, n_images, batch_size):
        actual_batch_size = min(batch_size, n_images - idx)

        # Load image batch
        image_batch = [
            ds.image.sel(image_id=idx + i).values
            for i in range(actual_batch_size)
        ]
        predictor.set_image_batch(image_batch)

        # Build boxes batch
        boxes_batch = [
            map_image_id_to_boxes[idx + i] for i in range(actual_batch_size)
        ]

        # Predict masks
        masks_batch, scores_batch, _ = predictor.predict_batch(
            box_batch=boxes_batch,
            multimask_output=False,
        )

        # Save boolean masks to zarr
        for i in range(actual_batch_size):
            masks_one_image = masks_batch[i].squeeze(axis=1)  # (N, H, W)
            n_objects = masks_one_image.shape[0]
            mask_zarr[idx + i, :n_objects] = masks_one_image

            print(
                f"Image {idx + i}: "
                f"{n_objects} masks / {boxes_batch[i].shape[0]} boxes"
            )

        # Free MPS/CUDA memory between images
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"Saved boolean masks to {zarr_path}")
else:
    # Download pre-computed masks
    # TODO: Replace with the actual URL and hash after generating
    # and hosting the pre-computed zarr store.
    precomputed_source = {
        "url": "https://example.com/waterfowl_masks.zarr.zip",
        "hash": None,
    }

    pooch.retrieve(
        url=precomputed_source["url"],
        known_hash=precomputed_source["hash"],
        fname="waterfowl_masks.zarr.zip",
        path=ethology_cache,
        processor=pooch.Unzip(extract_dir=str(tmp_dir)),
    )

    print("Downloaded pre-computed masks")


# %%
# Read masks and align with bounding boxes dataset
# --------------------------------------------------
#
# We read the masks from the zarr store and add them as a new variable
# in the bounding boxes dataset. Since the zarr array has shape
# ``(n_images, n_max_ids, img_h, img_w)`` and we use the same
# ``image_id`` and ``id`` dimensions, the masks are automatically
# aligned with the bounding box annotations.

mask_data = zarr.open(str(zarr_path), mode="r")

ds["mask"] = xr.DataArray(
    data=mask_data,
    dims=["image_id", "id", "img_h", "img_w"],
)

print(ds)
print(ds.sizes)

# %%
# We can now access masks using the same selectors as the bounding boxes.
# For example, ``ds.mask.sel(image_id=0, id=3)`` returns the boolean mask
# for annotation ID 3 in image 0.


# %%
# Visualise results
# ------------------
#
# Let's visualise the masks for one image. We overlay all masks on top
# of the image, and highlight one specific bounding box and its
# corresponding mask.

image_id = 0
annotation_id = 10

fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Show the image
ax.imshow(ds.image.sel(image_id=image_id).values)

# Plot centre of the selected bounding box
ax.scatter(
    ds.position.sel(image_id=image_id, id=annotation_id, space="x"),
    ds.position.sel(image_id=image_id, id=annotation_id, space="y"),
    s=15,
    marker="x",
    color="r",
)

# Plot the single mask for the selected annotation
single_mask = ds.mask.sel(image_id=image_id, id=annotation_id)
ax.imshow(
    single_mask,
    cmap="Blues",
    alpha=single_mask.astype(float) * 0.5,
)
ax.contour(single_mask, levels=[0.5], colors="red", linewidths=0.5)

# Overlay all masks in the image
all_masks = ds.mask.sel(image_id=image_id).any(dim="id")
ax.imshow(
    all_masks,
    cmap="turbo",
    alpha=all_masks.astype(float) * 0.5,
)

ax.set_title(f"Image {image_id} — all masks + annotation {annotation_id}")
ax.set_xlabel("x (pixels)")
ax.set_ylabel("y (pixels)")
plt.tight_layout()


# %%
# .. tip::
#    You can also visualise images and masks interactively in
#    `napari <https://napari.org>`_::
#
#       import napari
#
#       viewer = napari.Viewer()
#       viewer.add_image(
#           ds.image.values,
#           name="images",
#       )
#       viewer.add_labels(
#           ds.mask.values.astype(int).max(axis=1),
#           name="masks",
#       )


# %%
# Clean-up
# --------
# Remove the temporary directory containing the zarr store.

shutil.rmtree(tmp_dir)

# %%
