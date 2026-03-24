"""Compute

This example demos:
- how to run SAM2 to compute masks for a given bbox dataset
- how to store the output masks as zarr
- how to read the zarr store as a masks dataarray that we can align with our
  bboxes dataset
- how to visualise images and masks in napari

It also includes the ImageArray lazy loader and demos on how to
plot the data.

It uses a conditional fallback: run SAM2 if available, otherwise download pre-computed results
"""

# %%
from datetime import datetime
from pathlib import Path

import dask.array as da
import matplotlib.pyplot as plt
import napari
import numpy as np
import xarray as xr
import zarr

# from octron.sam_octron.helpers.sam2_zarr import (
#     create_image_zarr,
#     mark_frames_annotated,
# )
from PIL import Image
from sam2.sam2_image_predictor import SAM2ImagePredictor

from ethology.io.annotations import load_bboxes

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data

# Predictor
SAM_OCTRON_LOCAL_DIR = Path(
    "/Users/sofia/swc/project_octron/OCTRON-GUI/octron/sam_octron"
)
SAM2_CKPT_PATH = (
    SAM_OCTRON_LOCAL_DIR / "checkpoints" / "sam2.1_hiera_base_plus.pt"
)
SAM2_CONFIG_PATH = (
    SAM_OCTRON_LOCAL_DIR / "configs" / "sam2.1" / "sam2.1_hiera_b+.yaml"
)

# input bbox groundtruth data
DATA_DIR = Path("/Users/sofia/swc/CrabLabels/sep2023-full")
IMAGES_DIR = DATA_DIR / "frames"
ANNOTATIONS_FILE = DATA_DIR / "annotations" / "VIA_JSON_combined_coco_gen.json"

# Output dir
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path(
    f"/Users/sofia/arc/project_Zoo_crabs/crabs-exploration/output_{timestamp}"
)  # root output folder

# %%%%%%%%%%%%%%%%%%%%%%%%%%
# Helpers


class ImageArrayLazy:
    """A lazy array for images in a list."""

    def __init__(self, img_paths):
        self.img_paths = sorted(img_paths)
        # add image shape, assuming all have same as
        # first sample
        sample = np.array(Image.open(img_paths[0]))  # H, W, C
        self.img_h, self.img_w, self.img_c = sample.shape

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        return np.array(Image.open(self.img_paths[idx]))

    @property
    def shape(self):
        return (len(self.img_paths), self.img_h, self.img_w, self.img_c)
        # B, H, W, C


# %%%%%%%%%%%%%%%%%%%%%%
# Read groundtruth as ethology annotation dataset
ds_bboxes = load_bboxes.from_files(
    ANNOTATIONS_FILE,
    format="COCO",
    images_dirs=IMAGES_DIR,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load ground truth images as lazy array

# build lazy image array
png_files = sorted(ds_bboxes.attrs["images_directories"].glob("*.png"))
image_array = ImageArrayLazy(png_files)
print(image_array.shape)

# add as attribute?
# QUESTION: can I align it with xarray axes?
ds_bboxes.attrs["image_array"] = image_array


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load SAM2 predictor
# select device based on availability
# ....

# load impage predictor
image_predictor = SAM2ImagePredictor.from_pretrained(
    "facebook/sam2.1-hiera-base-plus", device="mps"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define per-frame exemplar bboxes from groundtruth data

# corner 1 is min x, min y
# corner 2 is max x, max y
x1y1 = ds_bboxes.position - ds_bboxes.shape / 2
x2y2 = ds_bboxes.position + ds_bboxes.shape / 2

# Each key is a frame index;
# value is an (N, 4) float32 array [x1, y1, x2, y2].
map_frame_idx_to_boxes = {
    idx: np.c_[
        x1y1.sel(image_id=idx).dropna(dim="id", how="all").values.T,
        x2y2.sel(image_id=idx).dropna(dim="id", how="all").values.T,
    ]
    for idx in range(len(ds_bboxes.image_id))
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%
# Initialise zarr store for output masks
data_id_str = DATA_DIR.name
output_masks_dir = OUTPUT_DIR / data_id_str
output_masks_dir.mkdir(parents=True, exist_ok=True)

# TODO: replace this OCTRON function by
# zarr directly
# TODO: zarr store to store arrays with dimensions
# (n_frames, n_ids, image_height,image_width) and fill value=0
mask_zarr = create_image_zarr(
    zarr_path=output_masks_dir / "masks.zarr",
    num_frames=ds_bboxes.attrs["image_array"].shape[0],
    image_height=ds_bboxes.attrs["image_array"].shape[1],
    image_width=ds_bboxes.attrs["image_array"].shape[2],
    fill_value=-1,
    dtype="int16",
    video_hash_abbrev=data_id_str,
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Process images in batches, predict boolean masks, save to zarr store
# TODO: save boolean mask in zarr store directly

batch_size = 2  # samples
n_frames = len(ds_bboxes.image_id)
img_h, img_w = ds_bboxes.attrs["image_array"].shape[1:3]

# loop thru images
for idx in range(0, n_frames, batch_size):
    # Adjust batch size if required
    actual_batch_size = min(batch_size, n_frames - idx)

    # Initialise id_mask for this batch
    # an integer array (B, H, W) where each pixel stores which object "owns" it
    # 0 = background, rest are 1-based object instance IDs
    id_mask_batch = np.zeros((actual_batch_size, img_h, img_w), dtype=np.int16)

    # Compute embeddings for image batch
    image_batch = [image_array[idx + i] for i in range(actual_batch_size)]
    image_predictor.set_image_batch(image_batch)

    # Compute list of of bboxes — list of (N_i, 4) arrays, one per image
    boxes_batch = [
        map_frame_idx_to_boxes[f_i]
        for f_i in range(idx, idx + actual_batch_size)
    ]

    # Predict batch of masks
    # masks_batch is a list — one (N, 1, H, W) array per image
    # where N is number of boxes
    masks_batch, scores_batch, _ = image_predictor.predict_batch(
        box_batch=boxes_batch,
        multimask_output=False,
    )

    # TODO: keep boolean masks directly and save to zarr store
    # Convert boolean masks to ID-encoded masks OCTRON expects
    # (higher ID wins in overlap)
    for idx_rel_batch in range(actual_batch_size):
        # Get masks for one frame
        masks_one_frame = masks_batch[idx_rel_batch].squeeze(
            axis=1
        )  # (N, H, W)

        # Convert boolean mask to 1-based integer mask per object ID
        n_objects = masks_one_frame.shape[0]
        obj_ids = np.arange(1, n_objects + 1, dtype=np.int16)[:, None, None]
        id_mask_batch[idx_rel_batch] = (masks_one_frame * obj_ids).max(axis=0)

        print(
            f"Frame {idx + idx_rel_batch}: "
            f"{n_objects} masks / {boxes_batch[idx_rel_batch].shape[0]} boxes"
        )

    # Save id_mask to zarr store
    mask_zarr[idx : idx + actual_batch_size] = id_mask_batch

    # # Mark frames as annotated in zarr store attributes
    # for frame_idx in range(idx, idx + actual_batch_size):
    #     mark_frames_annotated(mask_zarr, frame_idx)

print(f"Saved ID-encoded mask zarr to {output_masks_dir / 'masks.zarr'}")

# %%%%%%%%%%%%%%%%%%%%%%%
# Load masks from zarr store
zarr_root = zarr.open(
    output_masks_dir / "masks.zarr",
    mode="r",
)

mask_da_array = da.from_zarr(zarr_root["masks"])


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Add masks to bboxes ds with aligned ids

# TODO: ZARR STORE now has these dimensions
# (n_frames, n_ids, image_height,image_width), so we can
# directly do:

ds_bboxes["mask"] = xr.DataArray(
    data=mask_da_array,
    dims=["image_id", "id", "img_h", "img_w"],
)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Demo usage

# Select a sample image and a sample id
image_id = 40
id = 10

fig, ax = plt.subplots(1, 1)
# image
ax.imshow(ds_bboxes.image_array[image_id])

# plot centre of selected bbox
ax.scatter(
    ds_bboxes.position.sel(image_id=image_id, id=id, space="x"),
    ds_bboxes.position.sel(image_id=image_id, id=id, space="y"),
    15,
    marker="x",
    color="r",
)
# plot single mask
single_mask = ds_bboxes.mask.sel(image_id=image_id, id=id)
ax.imshow(
    single_mask,
    cmap="Blues",
    alpha=single_mask.astype(float) * 0.5,
)
ax.contour(single_mask, levels=[0.5], colors="red", linewidths=0.5)

# all masks in one frame (boolean masks, all will show in same color!)
all_masks = ds_bboxes.mask.sel(image_id=image_id).any(dim="id")
ax.imshow(
    all_masks,
    cmap="turbo",
    alpha=all_masks.astype(float) * 0.5,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load frames and masks in napari viewer

# TODO: use mask array in ds_bboxes instead

viewer = napari.Viewer()
# viewer.add_image(np.asarray(image_array).moveaxis(0, -1, 1, 2), name="image")
viewer.add_labels(np.asarray(mask_da_array), name=f"{LABEL_NAME} masks")
