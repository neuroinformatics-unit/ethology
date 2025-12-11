# %%
from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
from lightning import Trainer
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from ethology.detectors.models import SingleDetector
from ethology.io.annotations import load_bboxes, save_bboxes

# %%
# To use TF32 instead of full FP32
# torch.set_float32_matmul_precision('medium')  # or 'high'


# Note:
# Since we are using precision="16-mixed", most of the operations are already
# in FP16 and hitting the Tensor Cores. The set_float32_matmul_precision call
# would only affect the few remaining FP32 operations.
# %%
# Helpers
class SimpleImageDataset(Dataset):
    """A simple dataset for images with no ground-truth.

    It returns a dummy annotations dictionary.
    """

    def __init__(self, root_dir, file_pattern, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted(self.root_dir.glob(file_pattern))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = Path(self.root_dir) / self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, {}  # return a dummy annotations dict


# %%
# Input data
images_dir = Path(
    "/home/sminano/swc/project_ethology/07.09.2023-frames/Sep2023_day4_reencoded"
)

# %%
# Define model config
config = {
    "model_kwargs": {
        "num_classes": 2,
        "weights": None,
        "weights_backbone": None,
    },
    "checkpoint": "/home/sminano/swc/project_crabs/ml-runs/617393114420881798/f348d9d196934073bece1b877cbc4d38/checkpoints/last.ckpt",
}

# %%
# Define dataset

inference_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

dataset = SimpleImageDataset(
    images_dir,
    "*.png",
    transform=inference_transforms,
)

# dataloader
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
# Run inference using model on dataset
# (format as detections dataset)


# Instantiate detector
model = SingleDetector(config)

# Define trainer
trainer = Trainer(
    accelerator="gpu",
    devices=1,
    logger=False,
    precision="16-mixed",  # --- results change;
    # uses FP16 for most operations, FP32 for sensitive ones
    # This setting reduces memory and speeds up training
)

# dataset attrs
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

predictions_ds = model.run_inference(trainer, dataloader, attrs=ds_attrs)


# %%
# Alternative
# # TODO: can I do this at the end of the predict epoch?
# predictions = trainer.predict(model, dataloader)
# predictions_ds = model.format_predictions(
#     predictions,
#     {
#         "images_dir": images_dir,
#         "map_image_id_to_filename": {
#             id: filename.relative_to(
#                 "/home/sminano/swc/project_ethology/07.09.2023-frames"
#             )
#             for id, filename in enumerate(dataset.image_files)
#         },
#         "map_category_to_str": {1: "crab"},
#     },
# )

# %%
# Export as COCO annotations?
out_file = save_bboxes.to_COCO_file(predictions_ds, output_filepath="out.json")
# %%
# Load proofread annotations
proofread_ds = load_bboxes.from_files(
    "via_project_11Dec2025_12h44m_coco.json", format="COCO"
)

# %%
