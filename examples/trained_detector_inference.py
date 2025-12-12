"""Run inference with a trained detector
========================================

Run inference with a trained detector on a dataset of images for proofreading.
"""

# %%
# This example demonstrates how to run inference with a trained detector on a
# dataset of images for later proofreading using the VIA annotation tool.
#
# The example assumes that the detector has already been trained and saved as a
# checkpoint.
#
# The example also assumes that the dataset of images is stored in a directory.
# %%
# Imports
# -------
import os
from pathlib import Path

from lightning import Trainer
from torch.utils.data import DataLoader

from ethology.datasets.inference import (
    InferenceImageDataset,
    get_default_inference_transforms,
)
from ethology.detectors.models import SingleDetector
from ethology.io.annotations import save_bboxes

# For interactive plots: install ipympl with `pip install ipympl` and uncomment
# the following line in your notebook
# %matplotlib widget

# %%
# Precision settings
# ------------------
# To use TF32 instead of full FP32
# torch.set_float32_matmul_precision('medium')  # or 'high'

# Note:
# Since we are using precision="16-mixed", most of the operations are already
# in FP16 and hitting the Tensor Cores. The set_float32_matmul_precision call
# would only affect the few remaining FP32 operations.


# %%
# Input data
# -----------

# TODO:Fetch images for inference
images_dir = Path(
    "/home/sminano/swc/project_ethology/07.09.2023-frames/Sep2023_day4_reencoded"
)

# TODO:Fetch model config
config = {
    "model_class": "fasterrcnn_resnet50_fpn_v2",  # required
    "model_kwargs": {
        "num_classes": 2,  # required!
        "weights": None,
        "weights_backbone": None,
    },
    # required
    "checkpoint": (
        "/home/sminano/swc/project_crabs/ml-runs/"
        "617393114420881798/f348d9d196934073bece1b877cbc4d38/checkpoints/last.ckpt"
    ),
}

# %%
# Prepare dataset for inference
# -----------------------------

# Create dataset
dataset = InferenceImageDataset(
    images_dir,
    "*.png",
    transform=get_default_inference_transforms(),
)

# Create dataloader
dataloader = DataLoader(
    dataset,
    batch_size=12,  # 12,
    shuffle=False,
    num_workers=8,  # 4
    # collate_fn=collate_fn_varying_n_bboxes,
    # persistent_workers=True,
    # pin_memory=True,  # <-- Faster CPU->GPU transfer
    # because we guarantee a physical address for the data
    # in memory, so we can use DMA that directly takes it to
    # the GPU
    # prefetch_factor=4,  # <-- Prefetch more batches
    # multiprocessing_context="fork"
    # if ref_config["num_workers"] > 0 and torch.backends.mps.is_available()
    # else None,  # see https://github.com/pytorch/pytorch/issues/87688
)

# %%
# Prepare model and trainer
# -------------------------

# Instantiate detector
model = SingleDetector(config)
print(config)

# Instantiate trainer
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    logger=False,
    precision="16-mixed",
    # uses FP16 for most operations, FP32 for sensitive ones
    # This setting reduces memory and speeds up training
)

# Define dataset attrs to add to predictions (optional)
ds_attrs = {
    "images_dir": images_dir,
    "map_image_id_to_filename": {
        id: filename.relative_to(
            "/home/sminano/swc/project_ethology/07.09.2023-frames"
        )
        for id, filename in enumerate(dataset.image_files)
    },
    "map_category_to_str": {1: "crab"},
}

# %%
# Run inference using model on dataloader
# ---------------------------------------
# The predictions are formatted as an ``ethology`` detections dataset.

# Run inference using model on dataloader
predictions_ds = model.run_inference(trainer, dataloader, attrs=ds_attrs)


# %%
# Export predictions as COCO annotations for proofreading
# ---------------------------------------------------------


out_file = save_bboxes.to_COCO_file(predictions_ds, output_filepath="out.json")

# %%
# Load proofread annotations and compare
# ---------------------------------------
# proofread_ds = load_bboxes.from_files(
#     "via_project_11Dec2025_12h44m_coco.json", format="COCO"
# )

# Clean-up
# --------
# To remove the output files we have just created, we can run the following:

os.remove(out_file)
