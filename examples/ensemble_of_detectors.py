# %%
# imports

from pathlib import Path

import torch
import torchvision.transforms.v2 as transforms
import yaml
from lightning import Trainer
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

from ethology.detectors.ensembles.models import EnsembleDetector

# from ethology.detectors.evaluate import compute_precision_recall_ds
# from ethology.io.annotations import load_bboxes

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# Helper functions
def create_coco_dataset(
    images_dir: str | Path,
    annotations_file: str | Path,
    composed_transform: transforms.Compose,
) -> CocoDetection:
    """Create a COCO dataset for object detection.

    Note: transforms are applied to the full dataset. If the dataset
    is later split, all splits will have the same transforms.
    """
    dataset_coco = CocoDetection(
        root=images_dir,
        annFile=annotations_file,
        transforms=composed_transform,
    )

    # wrap dataset for transforms v2
    dataset_transformed = wrap_dataset_for_transforms_v2(dataset_coco)

    return dataset_transformed


def collate_fn_varying_n_bboxes(batch: tuple) -> tuple:
    """Collate function for dataloader with varying number of bounding boxes.

    A custom function is needed for detection
    because the number of bounding boxes varies
    between images of the same batch.
    See https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html#data-loading-and-training-loop

    Parameters
    ----------
    batch : tuple
        a tuple of 2 tuples, the first one holding all images in the batch,
        and the second one holding the corresponding annotations.

    Returns
    -------
    tuple
        a tuple of length = batch size, made up of (image, annotations)
        tuples.

    """
    return tuple(zip(*batch, strict=True))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data

dataset_dir = Path("/home/sminano/swc/project_crabs/data/aug2023-full")
annotations_dir = dataset_dir / "annotations"
annotations_file_path = annotations_dir / "VIA_JSON_combined_coco_gen.json"


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define a dataloader
# Define transforms for inference
inference_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

# Create COCO dataset
# TODO: convert from ethology detections dataset to COCO dataset
dataset_coco = create_coco_dataset(
    images_dir=Path(dataset_dir) / "frames",
    annotations_file=annotations_file_path,
    composed_transform=inference_transforms,
)

# dataloader
dataloader = DataLoader(
    dataset_coco,
    batch_size=12,
    shuffle=False,
    num_workers=4,
    collate_fn=collate_fn_varying_n_bboxes,
    persistent_workers=True,
    # multiprocessing_context="fork"
    # if ref_config["num_workers"] > 0 and torch.backends.mps.is_available()
    # else None,  # see https://github.com/pytorch/pytorch/issues/87688
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define a YAML config file for the ensemble of trained detectors

config = {
    "models": {
        "model_class": "fasterrcnn_resnet50_fpn_v2",
        # imported from torchvision.models.detection
        "model_kwargs": {
            "num_classes": 2,
            "weights": None,  # null in YAML becomes None in Python
            "weights_backbone": None,
        },
        "checkpoints": [
            "/home/sminano/swc/project_crabs/ml-runs/617393114420881798/f348d9d196934073bece1b877cbc4d38/checkpoints/last.ckpt",
            "/home/sminano/swc/project_crabs/ml-runs/617393114420881798/879d2f77e2b24adcb06b87d2fede6a04/checkpoints/last.ckpt",
        ],
    },
    "fusion": {
        "method": "wbf",
        "iou_th_ensemble": 0.5,
        "skip_box_th": 0.0001,
        "n_jobs": 2,  # workers for joblib.Parallel, n_workers should be <= number of CPU cores
        # "confidence_threshold_post_fusion": 0.0,
        # "max_n_detections": 300
    },
}
config_file = "ensemble_of_detectors.yaml"
with open(config_file, "w") as f:
    yaml.dump(config, f, sort_keys=False)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load the ensemble of detectors
ensemble_detector = EnsembleDetector(config_file)
print(f"Ensemble detector is on device: {ensemble_detector.device}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run the ensemble of detectors on a dataset
# Use Trainer for inference (this sets the device flexibly)
trainer = Trainer(accelerator="gpu", devices=1, logger=False)
raw_predictions = trainer.predict(ensemble_detector, dataloader)

# format predictions as ethology detections dataset
fused_detections_ds = ensemble_detector.format_predictions(raw_predictions)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# # Evaluate the ensemble model
# # - load ground truth
# # - compute metrics

# gt_bboxes_ds = load_bboxes.from_files(annotations_file_path, format="COCO")


# fused_detections_ds, gt_bboxes_ds = compute_precision_recall_ds(
#     pred_bboxes_ds=fused_detections_ds,
#     gt_bboxes_ds=gt_bboxes_ds,
#     iou_threshold=0.1,  # change to 0.5?
# )


# print(
#     "Ensemble model with confidence threshold post fusion: "
#     f"{ensemble_detector.config['fusion']['confidence_threshold_post_fusion']}"
# )
# print(f"Precision: {fused_detections_ds.precision.mean().values:.4f}")
# print(f"Recall: {fused_detections_ds.recall.mean().values:.4f}")
