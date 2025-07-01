"""Run detection on a Pytorch dataset and export results as a movement dataset.

A script to run detection only (no tracking) on a Pytorch dataset and
export the results in a format that can be loaded in movement napari widget.
"""

# %%
import ast
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import xarray as xr
from mean_average_precision import MetricBuilder
from mlflow.tracking import MlflowClient
from torch.utils.data import random_split
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set xarray options
xr.set_options(display_expand_attrs=False)

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

print(f"Using device: {device}")

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

# Sanitize bounding boxes?

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

    # Add to dict
    bboxes_xyxy = detections_dict["boxes"].cpu().numpy()

    detections_per_validation_sample[val_idx] = {
        "bbox_xyxy": bboxes_xyxy,
        "bbox_centroids": (bboxes_xyxy[:, 0:2] + bboxes_xyxy[:, 2:4]) / 2,
        "bbox_shapes": bboxes_xyxy[:, 2:4] - bboxes_xyxy[:, 0:2],
        "bbox_confidences": detections_dict["scores"].cpu().numpy(),
        "bbox_labels": detections_dict["labels"].cpu().numpy(),
    }


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute precision and recall per validation sample

pr_per_validation_sample = {}

iou_threshold = 0.1
recall_threshold = 0.0


# the mean average precision package assumes class_id starts at 0!
# so if there is only one class, it assumes its id is 0
metric_fn = MetricBuilder.build_evaluation_metric("map_2d", num_classes=1)

for idx_validation_sample in range(len(val_dataset)):
    # Get ground truth bboxes
    # [xmin, ymin, xmax, ymax, class_id, difficult, crowd]
    gt_bboxes_xyxy = np.c_[
        val_dataset[idx_validation_sample][1]["boxes"].cpu().numpy(),
        np.zeros(
            val_dataset[idx_validation_sample][1]["boxes"].shape[0]
        ),  # class_id = 0
        np.zeros(
            val_dataset[idx_validation_sample][1]["boxes"].shape[0]
        ),  # difficult
        np.zeros(
            val_dataset[idx_validation_sample][1]["boxes"].shape[0]
        ),  # crowd
    ]

    # Get predicted bboxes
    # make class_id 0-indexed!
    # [xmin, ymin, xmax, ymax, class_id, confidence]
    pred_bboxes_xyxy_conf = np.c_[
        detections_per_validation_sample[idx_validation_sample]["bbox_xyxy"],
        detections_per_validation_sample[idx_validation_sample]["bbox_labels"]
        - 1,  # class_id is 0-indexed!
        detections_per_validation_sample[idx_validation_sample][
            "bbox_confidences"
        ],
    ]

    # Add gt and pred bboxes to metric
    metric_fn.reset()
    metric_fn.add(pred_bboxes_xyxy_conf, gt_bboxes_xyxy)
    metric = metric_fn.value(
        iou_thresholds=[iou_threshold],
        recall_thresholds=np.array([recall_threshold]),
        mpolicy="soft",
    )

    # compute precision and recall for one frame
    pr_per_validation_sample[idx_validation_sample] = {
        "precision": metric[iou_threshold][0]["precision"][-1],
        "recall": metric[iou_threshold][0]["recall"][-1],
    }

# average across validation samples
print(
    f"Average precision: {
        np.mean([pr['precision'] for pr in pr_per_validation_sample.values()])
    }"
)
print(
    f"Average recall: {
        np.mean([pr['recall'] for pr in pr_per_validation_sample.values()])
    }"
)

# %%
# plot gt and pred bboxes for one frame (idx_validation_sample)

fig, ax = plt.subplots()
ax.imshow(val_dataset[idx_validation_sample][0].permute(1, 2, 0))
for i in range(gt_bboxes_xyxy.shape[0]):
    ax.add_patch(
        plt.Rectangle(
            (gt_bboxes_xyxy[i, 0], gt_bboxes_xyxy[i, 1]),
            gt_bboxes_xyxy[i, 2] - gt_bboxes_xyxy[i, 0],  # width
            gt_bboxes_xyxy[i, 3] - gt_bboxes_xyxy[i, 1],  # height
            fill=False,
            edgecolor=(0, 1, 0),
            linewidth=2.5,
        )
    )
for i in range(pred_bboxes_xyxy_conf.shape[0]):
    ax.add_patch(
        plt.Rectangle(
            (pred_bboxes_xyxy_conf[i, 0], pred_bboxes_xyxy_conf[i, 1]),
            pred_bboxes_xyxy_conf[i, 2] - pred_bboxes_xyxy_conf[i, 0],
            pred_bboxes_xyxy_conf[i, 3] - pred_bboxes_xyxy_conf[i, 1],
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
    )
plt.show()


# %%
# plot precision and recall for one iou threshold and last frame
plt.plot(
    metric[iou_threshold][0]["recall"],
    metric[iou_threshold][0]["precision"],
    ".-",
)
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.show()
