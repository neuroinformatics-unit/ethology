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
from ethology.detectors.evaluate import compute_precision_recall_ds
from ethology.io.annotations import load_bboxes

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
experiment_ID = "617393114420881798"
ml_runs_experiment_dir = (
    Path("/home/sminano/swc/project_crabs/ml-runs") / experiment_ID
)
last_ckpt = Path("checkpoints") / "last.ckpt"

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
            str(
                ml_runs_experiment_dir
                / "f348d9d196934073bece1b877cbc4d38"
                / last_ckpt
            ),  # above_0th
            str(
                ml_runs_experiment_dir
                / "879d2f77e2b24adcb06b87d2fede6a04"
                / last_ckpt
            ),  # above_1st
            str(
                ml_runs_experiment_dir
                / "75583ec227e3444ab692b99c64795325"
                / last_ckpt
            ),  # above_5th
            str(
                ml_runs_experiment_dir
                / "4acc37206b1e4f679d535c837bee2c2f"
                / last_ckpt
            ),  # above_10th
            str(
                ml_runs_experiment_dir
                / "fdcf88fcbcc84fbeb94b45ca6b6f8914"
                / last_ckpt
            ),  # above_25th
            str(
                ml_runs_experiment_dir
                / "daa05ded0ea047388c9134bf044061c5"
                / last_ckpt
            ),  # above_50th
        ],
    },
    "fusion": {
        "method": "wbf",
        "iou_th_ensemble": 0.5,
        "skip_box_th": 0.0001,
        "n_jobs": -1,  # workers for joblib.Parallel, n_workers should be <= number of CPU cores
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
# Remove low confidence detections
confidence_threshold_post_fusion = 0.5
fused_detections_ds_ = fused_detections_ds.where(
    fused_detections_ds.confidence >= confidence_threshold_post_fusion
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Evaluate the ensemble model
# - load ground truth
# - compute metrics

gt_bboxes_ds = load_bboxes.from_files(annotations_file_path, format="COCO")

iou_threshold_tp = 0.25
fused_detections_ds, gt_bboxes_ds = compute_precision_recall_ds(
    pred_bboxes_ds=fused_detections_ds_,
    gt_bboxes_ds=gt_bboxes_ds,
    iou_threshold=iou_threshold_tp,
)

# All models on full August dataset, without removing low confidence detections:
# confidence_threshold_post_fusion = 0.0
# Precision: 0.5920
# Recall: 0.8455
# ---
# confidence_threshold_post_fusion = 0.4
# Precision: 0.8339
# Recall: 0.7177
# ---
# confidence_threshold_post_fusion = 0.5
# Precision: 0.8714
# Recall: 0.6624
# ---

print(
    "Ensemble model with confidence threshold post fusion: "
    f"{confidence_threshold_post_fusion:.2f}"
)
print(f"Precision: {fused_detections_ds.precision.mean().values:.4f}")
print(f"Recall: {fused_detections_ds.recall.mean().values:.4f}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot calibration curve
