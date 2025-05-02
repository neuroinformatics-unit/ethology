"""Run detection only.

A script to run detection only and export them in a format that
can be loaded in movement napari widget
"""

# %%
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import yaml
from movement.io import load_poses, save_poses
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set default device: CUDA if available, otherwise mps, otherwise CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data
video_path = Path(
    "/home/sminano/swc/project_ethology/tap_models_crabs/input/04.09.2023-04-Right_RE_test.mp4"
)

trained_model_path = Path(
    "/home/sminano/swc/project_ethology/run_slurm_5313275_0/ml-runs_317777717624044570_40b1688a76d94bd08175cb380d0a6e0e_checkpoints_last.ckpt"
)
trained_model_config_path = Path(
    "/home/sminano/swc/project_ethology/run_slurm_5313275_0/01_config_all_data_augmentation.yaml"
)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Helper functions


def open_video(video_path: str | Path) -> cv2.VideoCapture:
    """Open video file."""
    video_object = cv2.VideoCapture(video_path)
    if not video_object.isOpened():
        raise Exception("Error opening video file")
    return video_object


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Load model

# Read config
with open(trained_model_config_path) as f:
    trained_model_config = yaml.safe_load(f)

# Load structure
model = fasterrcnn_resnet50_fpn_v2(
    weights=None,
    weights_backbone=None,
    num_classes=trained_model_config["num_classes"],
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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run detection only

# Initialise dict to store tracked bboxes
detections_all_frames = {}

# Loop over frames
frame_idx = 0
input_video_object = open_video(video_path)
total_n_frames = int(input_video_object.get(cv2.CAP_PROP_FRAME_COUNT))


while input_video_object.isOpened():
    # Read frame
    ret, frame = input_video_object.read()

    # If ret is False, it means we have reached the end of the video
    if not ret:
        break

    # Apply transforms to frame and place tensor on device
    image_tensor = inference_transforms(frame).to(device)[None]

    # Run detection
    with torch.no_grad():
        # use [0] to select the one image in the batch
        # Returns: dictionary with data of the predicted bounding boxes.
        # The keys are: "boxes", "scores", and "labels". The labels
        # refer to the class of the object detected, and not its ID.
        detections_dict = model(image_tensor)[0]

    # Add to dict
    bboxes_xyxy = detections_dict["boxes"].cpu().numpy()
    bbox_confidences = detections_dict["scores"].cpu().numpy()
    bbox_centroids = (bboxes_xyxy[:, 0:2] + bboxes_xyxy[:, 2:4]) / 2

    detections_all_frames[frame_idx] = {
        "bbox_centroids": bbox_centroids,  # detection_idx, x, y
        "bbox_confidences": bbox_confidences,  # detection_idx, confidence
    }

    # Update frame index
    frame_idx += 1


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Format detections as a movement dataset

max_detections_per_frame = max(
    [
        dets["bbox_centroids"].shape[0]
        for dets in detections_all_frames.values()
    ]
)
n_keypoints = 1

# Pad arrays with nans
position_array = np.full(
    (total_n_frames, 2, n_keypoints, max_detections_per_frame),
    np.nan,
)  # (n_frames, n_space, n_keypoints, n_individuals)
confidence_array = np.full(
    (total_n_frames, n_keypoints, max_detections_per_frame),
    np.nan,
)  # (n_frames, n_keypoints, n_individuals)
for frame_idx, dets in detections_all_frames.items():
    position_array[frame_idx, :, :, : dets["bbox_centroids"].shape[0]] = (
        np.transpose(dets["bbox_centroids"][None], (-1, 0, 1))
    )
    confidence_array[frame_idx, :, : dets["bbox_centroids"].shape[0]] = dets[
        "bbox_confidences"
    ][None, None]


# %%
ds = load_poses.from_numpy(
    position_array=position_array,
    confidence_array=confidence_array,
    individual_names=[
        f"untracked_{i}" for i in range(max_detections_per_frame)
    ],
    keypoint_names=["centroid"],
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Export movement dataset

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_poses.to_sleap_analysis_file(
    ds, f"detections_untracked_{video_path.stem}_{timestamp}.h5"
)

# %%
