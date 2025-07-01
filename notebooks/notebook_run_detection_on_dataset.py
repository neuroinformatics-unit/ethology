"""Run detection on a Pytorch dataset and export results as a movement dataset.

A script to run detection only (no tracking) on a Pytorch dataset and
export the results in a format that can be loaded in movement napari widget.
"""

# %%
import ast
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
import xarray as xr
from mlflow.tracking import MlflowClient
from pycocotools.coco import COCO
from torch.utils.data import random_split
from torchvision.datasets import CocoDetection, wrap_dataset_for_transforms_v2
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

from ethology.annotations.io import save_bboxes

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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Format detections as an ethology detections dataset
# (validate as ethology annotations dataset? or add from_numpy?)

# Get params for array dimensions
max_detections_per_frame = max(
    [
        dets["bbox_centroids"].shape[0]
        for dets in detections_per_validation_sample.values()
    ]
)
n_keypoints = 1
total_n_frames = len(val_dataset)

# Initialise position, shape and label arrays
array_dict = {}
array_dict["position_array"] = np.full(
    (total_n_frames, 2, max_detections_per_frame),
    np.nan,
)  # (n_frames, n_space, n_individuals)
array_dict["shape_array"] = np.full(
    (total_n_frames, 2, max_detections_per_frame),
    np.nan,
)  # (n_frames, n_space, n_individuals)
array_dict["category_array"] = np.full(
    (total_n_frames, max_detections_per_frame),
    -1,  # -1 is the default value for missing data
)  # (n_frames, n_individuals)
array_dict["score_array"] = np.full(
    (total_n_frames, max_detections_per_frame),
    np.nan,
)  # (n_frames, n_individuals)

# Fill in values
for frame_idx, dets in detections_per_validation_sample.items():
    array_dict["position_array"][
        frame_idx, :, : dets["bbox_centroids"].shape[0]
    ] = np.transpose(dets["bbox_centroids"])
    array_dict["shape_array"][frame_idx, :, : dets["bbox_shapes"].shape[0]] = (
        np.transpose(dets["bbox_shapes"])
    )
    array_dict["category_array"][frame_idx, : dets["bbox_labels"].shape[0]] = (
        dets["bbox_labels"]
    )
    array_dict["score_array"][
        frame_idx, : dets["bbox_confidences"].shape[0]
    ] = dets["bbox_confidences"]

# Format detections on validation set as ethology detections dataset
# (detections dataset is a like ethology annotations dataset but with
# confidence scores)
ds = xr.Dataset(
    data_vars=dict(
        position=(
            ["image_id", "space", "id"],
            array_dict["position_array"],
        ),
        shape=(["image_id", "space", "id"], array_dict["shape_array"]),
        category=(["image_id", "id"], array_dict["category_array"]),
        confidence=(["image_id", "id"], array_dict["score_array"]),
    ),
    coords=dict(
        # use image_id from ground truth annotations!
        image_id=[
            val_dataset[i][1]["image_id"] for i in range(total_n_frames)
        ],
        space=["x", "y"],
        id=range(max_detections_per_frame),
        # annotation ID per frame; could be consistent across frames
        # or not
    ),
)

print(ds)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get map from image_id to filename from ground truth annotations

cocoGt = COCO(str(dataset_dir / "annotations/VIA_JSON_combined_coco_gen.json"))

# compute map from image_id to filename
# assuming val_dataset[0][1]['image_id'] is the image_id of the first frame
# in the validation set, for the image IDs as specified in the ground truth annotation
# file
df_images = pd.DataFrame(cocoGt.dataset["images"])
map_image_id_to_filename_gt = {
    image_id: filename
    for image_id, filename in zip(
        df_images["id"], df_images["file_name"], strict=False
    )
}

# # map image_id in xarray to filename
# map_val_image_id_to_filename = {
#     idx: map_image_id_to_filename_gt[val_dataset[idx][1]["image_id"]]
#     for idx in range(total_n_frames)
# }

# compute map from category_id to category_name
df_categories = pd.DataFrame(cocoGt.dataset["categories"])
map_category_id_to_category_name = {
    category_id: category_name
    for category_id, category_name in zip(
        df_categories["id"],
        df_categories["name"],
        strict=True,
    )
}

# add map to ds
ds.attrs["map_image_id_to_filename"] = map_image_id_to_filename_gt
ds.attrs["map_category_id_to_category"] = map_category_id_to_category_name

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export detections on validation set as COCO JSON file

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = Path(
    f"{mlflow_params['run_name']}_detections_val_set_{timestamp}.json"
)
out_path = save_bboxes.to_COCO_file(
    ds,
    output_parent_dir / filename,
)

# Note: this is not an official COCO format for results
# The format for annotations and detections is different
# https://cocodataset.org/#format-results
