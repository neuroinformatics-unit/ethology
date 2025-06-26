"""Run detection on a Pytorch dataset and export results as a movement dataset.

A script to run detection only (no tracking) on a Pytorch dataset and
export the results in a format that can be loaded in movement napari widget.
"""

# %%
import ast
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
from mlflow.tracking import MlflowClient
from movement.io import load_poses, save_poses
from torch.utils.data import random_split
from torchmetrics.detection import MeanAveragePrecision
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
dataset_dir = Path("/home/sminano/swc/project_crabs/data/sep2023-full")

trained_model_path = Path(
    "/home/sminano/swc/project_crabs/ml-runs/617393114420881798/f348d9d196934073bece1b877cbc4d38/checkpoints/last.ckpt"
)

trained_model_mlflow_params_path = Path(
    "/home/sminano/swc/project_crabs/ml-runs/617393114420881798/f348d9d196934073bece1b877cbc4d38/params"
)  # for config


# to save output frames and detections
output_parent_dir = Path("/home/sminano/swc/project_ethology")

flag_save_frames = False

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set default device: CUDA if available, otherwise mps, otherwise CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Retrieve model config and CLI args from mlflow


def read_mlflow_params(
    trained_model_path: str, tracking_uri: str = None
) -> dict:
    """Read parameters for a specific MLflow run."""
    # Create MLflow client
    mlruns_path = str(Path(trained_model_path).parents[3])
    client = MlflowClient(tracking_uri=mlruns_path)

    # Get the run
    runID = Path(trained_model_path).parents[1].stem
    run = client.get_run(runID)

    # Access parameters
    params = run.data.params
    params["run_name"] = run.info.run_name

    return params


mlflow_params = read_mlflow_params(trained_model_path)
config = {
    k.removeprefix("config/"): ast.literal_eval(v)
    for k, v in mlflow_params.items()
    if k.startswith("config/")
}


def safe_eval_string(s):
    """Try to evaluate a string as a literal, otherwise return as-is."""
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        # return as-is if not a valid literal
        return s


cli_args = {
    k.removeprefix("cli_args/"): safe_eval_string(v)
    for k, v in mlflow_params.items()
    if k.startswith("cli_args/")
}

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load model

# Load structure
model = fasterrcnn_resnet50_fpn_v2(
    weights=None,
    weights_backbone=None,
    num_classes=config["num_classes"],
)

# Read state dict
state_dict = torch.load(trained_model_path)
state_dict_model = {
    k.lstrip("model."): v
    for k, v in state_dict["state_dict"].items()
    if k.startswith("model.")
}

# Load weights into model and set to evaluation mode
model.load_state_dict(state_dict_model)
model.eval()
model.to(device)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define transforms to apply to input frames
inference_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Build Pytorch dataset
seed_n = cli_args["seed_n"]
annotations_filename = Path(cli_args["annotation_files"][0]).name

# create "default" COCO dataset
dataset_coco = CocoDetection(
    Path(dataset_dir) / "frames",
    Path(dataset_dir) / "annotations" / annotations_filename,
    transforms=inference_transforms,
)

# wrap dataset for transforms v2
dataset_transformed = wrap_dataset_for_transforms_v2(dataset_coco)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Split dataset
# def _collate_fn(self, batch: tuple) -> tuple:
#     """Collate function used for dataloaders.

#     A custom function is needed for detection
#     because the number of bounding boxes varies
#     between images of the same batch.
#     See https://pytorch.org/vision/main/auto_examples/transforms/plot_transforms_e2e.html#data-loading-and-training-loop

#     Parameters
#     ----------
#     batch : tuple
#         a tuple of 2 tuples, the first one holding all images in the batch,
#         and the second one holding the corresponding annotations.

#     Returns
#     -------
#     tuple
#         a tuple of length = batch size, made up of (image, annotations)
#         tuples.

#     """
#     return tuple(zip(*batch))


# Split data into train and test-val sets
rng_train_split = torch.Generator().manual_seed(seed_n)
rng_val_split = torch.Generator().manual_seed(seed_n)

train_dataset, test_val_dataset = random_split(
    dataset_transformed,
    [config["train_fraction"], 1 - config["train_fraction"]],
    generator=rng_train_split,
)

# Split test/val sets from the remainder
test_dataset, val_dataset = random_split(
    test_val_dataset,
    [
        1 - config["val_over_test_fraction"],
        config["val_over_test_fraction"],
    ],
    generator=rng_val_split,
)

print(f"Seed: {seed_n}")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")
print(f"Number of test samples: {len(test_dataset)}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run detection on validation set

# TODO: use dataloader for efficiency?
detections_per_validation_sample = {}
annotations_per_validation_sample = {}

# # initialise metric
# metric = MeanAveragePrecision(
#     box_format="xyxy",
#     iou_type="bbox",
#     iou_thresholds=None,#[0.1],  # If set to None [0.5,...,0.95] with step 0.05
#     rec_thresholds=None,  # If set to None [0,...,1] with step 0.01
#     max_detection_thresholds=[1, 100, 1000],
#     extended_summary=True,
#     average="micro",  # macro=average per class first, should be same for me?
# )

metric_per_frame = MeanAveragePrecision(
    box_format="xyxy",
    iou_type="bbox",
    iou_thresholds=[0.1], # 0.5  # If set to None [0.5,...,0.95] with step 0.05
    rec_thresholds=None,  # If set to None [0,...,1] with step 0.01 -- these are the interpolation points
    max_detection_thresholds=[10, 100, 1000],
    extended_summary=True,
    average="micro",  # macro=average per class first, should be same for me?
)

# create output directory if it doesn't exist
if flag_save_frames:
    output_dir = output_parent_dir / f"{dataset_dir.name}_val_seed_n_{seed_n}"
    output_dir.mkdir(parents=True, exist_ok=True)


recall_per_sample = []
precision_per_sample = []
map_per_sample = []

for val_idx, (image, annotations) in enumerate(val_dataset):
    # Apply transforms to frame and place tensor on device
    image_tensor = inference_transforms(image).to(device)[None]

    # Put annotations in same device as image
    annotations["boxes"] = annotations["boxes"].to(device)
    annotations["labels"] = annotations["labels"].to(device)

    # Run detection
    with torch.no_grad():
        # use [0] to select the one image in the batch
        # Returns: dictionary with data of the predicted bounding boxes.
        # The keys are: "boxes", "scores", and "labels". The labels
        # refer to the class of the object detected, and not its ID.
        detections_dict = model(image_tensor)[0]

        # add to metric
        # metric.update([detections_dict], [annotations])

        # add to metric per frame
        metric_per_frame.reset()
        metric_per_frame.update([detections_dict], [annotations])
        metrics_one_frame = metric_per_frame.compute()

        # map is area under P-C curve, averaged over all IOU thresholds?
        recall_one_frame = metrics_one_frame["recall"][0, 0, 0, -1].item()

        precision_values = metrics_one_frame["precision"][
            0, :, 0, 0, -1
        ].numpy()  # first IOU threshold, first class, first area (?), max detections = 1000
        idcs_precision_non_zero = np.nonzero(precision_values)[0]
        precision_one_frame = precision_values[idcs_precision_non_zero[-1]]

        map_one_frame = metrics_one_frame["map"].item()

        # add to list
        recall_per_sample.append(recall_one_frame)
        precision_per_sample.append(precision_one_frame)
        map_per_sample.append(map_one_frame)
        

        print(
            f"Validation sample {val_idx}, "
            f"mAP: {map_one_frame}, "  # area under P-C curve, averaged over all IOU thresholds
            f"Precision: {precision_one_frame}, "
            f"Recall: {recall_one_frame}"
        )
        # recall for first IOU threshold, first class, first area (?),
        # max detections = 1000

    # Add to dict
    bboxes_xyxy = detections_dict["boxes"].cpu().numpy()
    bbox_confidences = detections_dict["scores"].cpu().numpy()
    bbox_centroids = (bboxes_xyxy[:, 0:2] + bboxes_xyxy[:, 2:4]) / 2

    detections_per_validation_sample[val_idx] = {
        "bbox_centroids": bbox_centroids,  # detection_idx, x, y
        "bbox_confidences": bbox_confidences,  # detection_idx, confidence
    }

    # add to dict
    # annotations_per_validation_sample[val_idx] = annotations

    # Save image
    if flag_save_frames:
        image_path = output_dir / f"frame_val_idx_{val_idx:06d}.png"
        image_array = (image.permute(1, 2, 0).numpy() * 255).astype(
            np.uint8
        )  # (C, H, W) -> (H, W, C)
        cv2.imwrite(
            image_path, cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        )  # cv2 assumes BGR


# metrics = metric.compute()
# print(metrics["map"])

print(f"Mean recall: {np.mean(recall_per_sample)}") 
# 0.8494677009613534 @ IOU=0.1
# 0.8033303880293905 @ IOU=0.5
print(f"Mean precision: {np.mean(precision_per_sample)}")  
# 0.9767496450305635 @ IOU=0.1
# 0.929829445017168 @ IOU=0.5

# %%
# Compute metrics
# metrics = metric.compute()

# print(metrics["map"])

# %%
# Show iou > threshold for detections in first image
# as a matrix with n_rows = n_detections, n_cols = n_gt_boxes

# import matplotlib.pyplot as plt

# # first image, first class
# plt.imshow(metrics['ious'][(np.int64(0), np.int64(1))] > 0.5)

# # 30th image, first class
# plt.imshow(metrics['ious'][(np.int64(30), np.int64(1))] > 0.5)


# %%
# P-R curve for first IOU threshold, first class, first area (?), max detections = 1000
import matplotlib.pyplot as plt

plt.scatter(
    x=np.arange(0, 1.01, 0.01),  # recall
    y=metrics_one_frame["precision"][0, :, 0, 0, -1],  # precision
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Format detections as a movement dataset

# Get params for array dimensions
max_detections_per_frame = max(
    [
        dets["bbox_centroids"].shape[0]
        for dets in detections_per_validation_sample.values()
    ]
)
n_keypoints = 1
total_n_frames = len(val_dataset)

# Initialise position and confidence arrays
position_array = np.full(
    (total_n_frames, 2, n_keypoints, max_detections_per_frame),
    np.nan,
)  # (n_frames, n_space, n_keypoints, n_individuals)
confidence_array = np.full(
    (total_n_frames, n_keypoints, max_detections_per_frame),
    np.nan,
)  # (n_frames, n_keypoints, n_individuals)

# Fill in values
for frame_idx, dets in detections_per_validation_sample.items():
    position_array[frame_idx, :, :, : dets["bbox_centroids"].shape[0]] = (
        np.transpose(dets["bbox_centroids"][None], (-1, 0, 1))
    )
    confidence_array[frame_idx, :, : dets["bbox_centroids"].shape[0]] = dets[
        "bbox_confidences"
    ][None, None]

# format as movement dataset
ds = load_poses.from_numpy(
    position_array=position_array,
    confidence_array=confidence_array,
    individual_names=[
        f"untracked_{i}" for i in range(max_detections_per_frame)
    ],
    keypoint_names=["centroid"],
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export movement dataset as .slp file
# in the future: export as VIA tracks file (after PR merged!)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = Path(
    f"{mlflow_params['run_name']}_detections_val_set_{timestamp}.h5"
)
save_poses.to_sleap_analysis_file(ds, output_parent_dir / filename)

# %%

# %%
