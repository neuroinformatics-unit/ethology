"""Run inference with a trained detector
========================================

Run inference with a trained detector on a dataset of images for proofreading.
"""

# %%
# This example demonstrates how to run inference with a trained detector on a
# dataset of images for later proofreading, for example using the VIA
# annotation tool.

# %%
# Imports
# -------
import os
from datetime import datetime
from pathlib import Path

import pooch
from lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights

from ethology.datasets.inference import (
    InferenceImageDataset,
    get_default_inference_transforms,
)
from ethology.detectors.models import ObjectDetector
from ethology.io.annotations import save_bboxes

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget


# %%
# Download dataset
# -----------------
# Source of the dataset
# African Wildlife Dataset
# url: https://github.com/ultralytics/assets/releases/download/v0.0.0/african-wildlife.zip
# AGPL license?
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
# Prepare dataset for inference
# -----------------------------

# Create dataset
images_dir = data_dir / "experts" / "images"
dataset = InferenceImageDataset(
    images_dir,
    "*.jpg",
    transforms=get_default_inference_transforms(),
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=12,  # 12,
    shuffle=False,
    num_workers=8,  # 4
)

# %%
# Prepare model and trainer
# -------------------------

# Pretreained options
# ObjectDetector({"model_class": "fcos_resnet50_fpn"})
# -- loads fcos_resnet50_fpn with coco2017 weights (91 classes)

# ObjectDetector({"model_class": "fcos_resnet50_fpn",
#   "model_kwargs":{"num_classes": 3}})
# -- fcos_resnet50_fpn with coco2017 weights except for the last
#    layers that are reinitialised with random weights for an output shape of
#    3 classes

# From checkpoint:
# ObjectDetector({"model_class": "fcos_resnet50_fpn",
#  "checkpoint": /path/to/ckpt})
# -- will complain if checkpoint does not have 91 classes

# Instantiate detector
# Define model config
# Faster-RCNN for 91 classes pretrained on COCO2017

detector = ObjectDetector(
    {
        "model_class": "fasterrcnn_resnet50_fpn_v2",
        # "model_kwargs": {"weights": "DEFAULT"}, # for pretrained
    }
)


# Instantiate trainer
trainer = Trainer(
    accelerator="cpu",  # recommended gpu if available
    devices=1,
    logger=False,
)

# %%
# Define dataset attrs to add to predictions
# -------------------------------------------

# We need to add `map_category_to_str` as a dataset attribute
# to be able to export the predictions as COCO

# %%
# Retrieve list of categories used in torch models trained on COCO2017
weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
list_category_str = weights.meta["categories"]


ds_attrs = {
    "images_dir": images_dir,
    "map_image_id_to_filename": {
        id: filename.relative_to(images_dir)
        for id, filename in enumerate(dataset.image_files)
    },
    "map_category_to_str": {k: cat for k, cat in enumerate(list_category_str)},
}  # required for COCO export

# %%
# Run inference using model on dataloader
# ----------------------------------------
# The predictions are formatted as an ``ethology`` detections dataset.

# Run inference using model on dataloader
predictions_ds = detector.run_inference(trainer, dataloader, attrs=ds_attrs)


# %%
# Export predictions as COCO annotations for proofreading
# ---------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = save_bboxes.to_COCO_file(
    predictions_ds, output_filepath=f"out_{timestamp}.json"
)

# %%
# Load proofread annotations and compare
# ---------------------------------------

# proofread_ds = load_bboxes.from_files(
#     "via_project_23Dec2025_15h25m_coco.json",
#     format="COCO",
# )

# %%
# Clean-up
# ---------
# To remove the output files we have just created, we can run the following:

os.remove(out_file)
