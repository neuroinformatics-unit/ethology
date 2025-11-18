# %%
# imports

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import xarray as xr
import yaml
from lightning import Trainer
from matplotlib import pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2

from ethology.detectors.ensembles.fusion import fuse_ensemble_detections
from ethology.detectors.ensembles.models import EnsembleDetector
from ethology.detectors.evaluate import compute_precision_recall_ds
from ethology.io.annotations import load_bboxes

# %%
# %matplotlib widget
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
images_dir = dataset_dir / "frames"
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
# gt_bboxes_ds = load_bboxes.from_files(annotations_file_path, format="COCO")
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
        "method": "weighted_boxes_fusion",
        # "nms", "soft_nms", "weighted_boxes_fusion" or "non_maximum_weighted"
        "method_kwargs": {  # arguments as in ensemble_boxes.weighted_boxes_fusion
            "iou_thr": 0.5,  # iou threshold for the ensemble
            "skip_box_thr": 0.0001,
        },
        # "n_jobs": -1,  # workers for joblib.Parallel, n_workers should be <= number of CPU cores
        # "confidence_threshold_post_fusion": 0.0,
        "max_n_detections": 300,
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
_ = trainer.predict(ensemble_detector, dataloader)


# Format predictions as ethology detections dataset and add attrs
# TODO: think about syntax of format_predictions (should it be instance or
# static method instead?)
# Q: Can it just be output from .predict?
# TODO: dataloader to ethology detections dataset
gt_bboxes_ds = load_bboxes.from_files(
    annotations_file_path, format="COCO", images_dirs=images_dir
)
ensemble_detections_ds = ensemble_detector.format_predictions(
    attrs=gt_bboxes_ds.attrs
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Some nice plots:
# ensemble_detections_ds.confidence.sel(image_id=0).plot()
# ensemble_detections_ds.confidence.sel(model=0).plot()
for m in range(5):
    plt.figure()
    ensemble_detections_ds.confidence.sel(model=m).plot()


# %%%%%%%%
# All models predict less boxes and have less avg confidence per image in
# image_ids from 350 to 450. Let's inspect video names and images for these
# samples.

# Add video name array
video_name = [
    ensemble_detections_ds.map_image_id_to_filename[img_id].split("_frame")[0]
    for img_id in ensemble_detections_ds.image_id.values
]
ensemble_detections_ds["video"] = xr.DataArray(video_name, dims="image_id")

# which videos?
np.unique(ensemble_detections_ds.video.sel(image_id=range(350, 450)).values)

# %%%%%%
# Visualise image
for image_id in range(350, 450, 10):
    image_filename = ensemble_detections_ds.map_image_id_to_filename[image_id]
    image_path = ensemble_detections_ds.images_directories / image_filename

    # img = Image.open(image_path)
    img = plt.imread(image_path)

    plt.figure()
    plt.imshow(img)
    plt.title(f"{image_filename}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fuse detections across models with WBF
# TODO: think whether joblib approach is more readable?
image_width_height = np.array(dataloader.dataset[0][0].shape[-2:])[::-1]
ensemble_detections_ds.attrs['image_shape'] = image_width_height

config_fusion = config["fusion"]


# %%
fused_detections_ds = fuse_ensemble_detections(
    ensemble_detections_ds,
    fusion_method=config_fusion['method'],
    fusion_method_kwargs=config_fusion["method_kwargs"],
    # max_n_detections=config_fusion["max_n_detections"],
    # should be larger than expected maximum number of detections after fusion
    # ---- method kwargs ----
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fuse detections across models with NMS

fused_detections_nms_ds = fuse_ensemble_detections(
    ensemble_detections_ds,
    fusion_method='soft_nms',
    fusion_method_kwargs={
        "iou_thr": config_fusion["method_kwargs"]["iou_thr"],
        "sigma":0.5,
        "thresh":0.001
    },
    max_n_detections=500
)

fused_detections_ds = fused_detections_nms_ds
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

# gt_bboxes_ds = load_bboxes.from_files(annotations_file_path, format="COCO")

iou_threshold_tp = 0.25
fused_detections_ds_, gt_bboxes_ds = compute_precision_recall_ds(
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
print(f"Precision: {fused_detections_ds_.precision.mean().values:.4f}")
print(f"Recall: {fused_detections_ds_.recall.mean().values:.4f}")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Plot calibration curve


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Evaluate single models
list_detections_ds_eval = []
for k in range(ensemble_detections_ds.sizes["model"]):
    detections_ds, _ = compute_precision_recall_ds(
        pred_bboxes_ds=ensemble_detections_ds.sel(model=k),
        gt_bboxes_ds=gt_bboxes_ds,
        iou_threshold=iou_threshold_tp,
    )
    list_detections_ds_eval.append(detections_ds)

    print(f"Model: {k}")
    print(f"Precision: {detections_ds.precision.mean().values:.4f}")
    print(f"Recall: {detections_ds.recall.mean().values:.4f}")
    print("--------------------------------")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualise detections
