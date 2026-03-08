# %%
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms.v2 as transforms
import yaml
from boxmot import BotSort
from movement.io import load_bboxes
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
    # "/home/sminano/swc/project_ethology/tap_models_crabs/input/04.09.2023-04-Right_RE_test.mp4"
    "/home/sminano/swc/project_ethology/boxmot/2_escape-birds-eye_many-crabs.mp4"
)

trained_model_path = Path(
    "/home/sminano/swc/project_ethology/detector-slurm_5313275_0/ml-runs_317777717624044570_40b1688a76d94bd08175cb380d0a6e0e_checkpoints_last.ckpt"
)
trained_model_config_path = Path(
    "/home/sminano/swc/project_ethology/detector-slurm_5313275_0/01_config_all_data_augmentation.yaml"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parameters
confidence_threshold = 0.5

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Helper functions


def open_video(video_path: str) -> cv2.VideoCapture:
    """Open video file."""
    video_object = cv2.VideoCapture(video_path)
    if not video_object.isOpened():
        raise Exception("Error opening video file")
    return video_object


def write_tracked_detections_to_csv(
    csv_file_path: str,
    tracked_bboxes_dict: dict,
    frame_name_regexp: str = "frame_{frame_idx:08d}.png",
    all_frames_size: int = 8888,
):
    """Write tracked detections to a csv file."""
    # Initialise csv file
    csv_file = open(csv_file_path, "w")
    csv_writer = csv.writer(csv_file)

    # write header following VIA convention
    # https://www.robots.ox.ac.uk/~vgg/software/via/docs/face_track_annotation.html
    csv_writer.writerow(
        (
            "filename",
            "file_size",
            "file_attributes",
            "region_count",
            "region_id",
            "region_shape_attributes",
            "region_attributes",
        )
    )

    # write detections
    # loop thru frames
    for frame_idx in tracked_bboxes_dict:
        # loop thru all boxes in frame
        for bbox, id, pred_score in zip(
            tracked_bboxes_dict[frame_idx]["tracked_boxes"],
            tracked_bboxes_dict[frame_idx]["ids"],
            tracked_bboxes_dict[frame_idx]["scores"],
            strict=False,
        ):
            # extract shape
            xmin, ymin, xmax, ymax = bbox
            width_box = int(xmax - xmin)
            height_box = int(ymax - ymin)

            # Add to csv
            csv_writer.writerow(
                (
                    frame_name_regexp.format(
                        frame_idx=frame_idx
                    ),  # f"frame_{frame_idx:08d}.png",  # frame index!
                    all_frames_size,  # frame size
                    '{{"clip":{}}}'.format("123"),
                    1,
                    0,
                    f'{{"name":"rect","x":{xmin},"y":{ymin},"width":{width_box},"height":{height_box}}}',
                    f'{{"track":"{int(id)}", "confidence":"{pred_score}"}}',
                )
            )


# def configure_model(config: dict) -> torch.nn.Module:
#     """Configure Faster R-CNN model.

#     Use default weights,
#     specified backbone, and box predictor.
#     """
#     model = fasterrcnn_resnet50_fpn_v2(weights="DEFAULT")
#     in_features = model.roi_heads.box_predictor.cls_score.in_features
#     model.roi_heads.box_predictor = faster_rcnn.FastRCNNPredictor(
#         in_features,
#         config["num_classes"],
#     )
#     return model


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
# Initialize the tracker
tracker = BotSort(
    reid_weights=Path("osnet_x0_25_msmt17.pt"),  # Path to ReID model
    device="0",  # why not device? why is this in GPU if we then copy to CPU?
    half=False,
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run detection and tracking


# Initialise dict to store tracked bboxes
tracked_detections_all_frames = {}

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

    # --------------------------------------------------------------------------
    # Run detection
    with torch.no_grad():
        # use [0] to select the one image in the batch
        # Returns: dictionary with data of the predicted bounding boxes.
        # The keys are: "boxes", "scores", and "labels". The labels
        # refer to the class of the object detected, and not its ID.
        detections_dict = model(image_tensor)[0]

    # Consider only the detections with a confidence greater than the threshold
    detections_list = []
    for i, score in enumerate(detections_dict["scores"]):
        if score >= confidence_threshold:
            bbox = detections_dict["boxes"][i].cpu().numpy()
            label = detections_dict["labels"][i].item()
            conf = score.item()
            detections_list.append([*bbox, conf, label])

    # Format detections for tracking
    # Convert detections to numpy array (N X (x, y, x, y, conf, cls))
    detections_array = np.array(detections_list)

    # --------------------------------------------------------------------------
    # Run tracker on detections

    # Update the tracker
    tracked_boxes_array = tracker.update(
        detections_array, frame
    )  # --> M X (x, y, x, y, id, conf, cls, ind)
    # ind is the index of the corresponding detection in the detections_array

    # Make it M X 5 (x, y, x, y, id)
    # tracked_boxes_array = tracked_boxes_array[:, :5]
    # Array of tracked bounding boxes with object IDs added as the last
    # column. The shape of the array is (n, 5), where n is the number of
    # tracked boxes. The columns correspond to the values (xmin, ymin,
    # xmax, ymax, id).

    # --------------------------------------------------------------------------
    # Add data to dict; key is frame index (0-based) for input clip
    tracked_detections_all_frames[frame_idx] = {
        "tracked_boxes": tracked_boxes_array[:, :4],  # :-1],  # (x, y, x, y)
        "ids": tracked_boxes_array[
            :, 4
        ],  # -1],  # IDs are the last(5th) column
        "scores": detections_dict["scores"],
    }

    # Update frame index
    frame_idx += 1

# Release video object
input_video_object.release()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Write tracked detections as VIA-tracks file
# to inspect in napari

# timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{video_path.stem}_tracked_detections_{timestamp}.csv"

write_tracked_detections_to_csv(
    csv_file_path=filename,
    tracked_bboxes_dict=tracked_detections_all_frames,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read VIA-tracks file as a movement dataset

ds = load_bboxes.from_via_tracks_file(
    file_path=filename, use_frame_numbers_from_file=False
)


# %%
# Check jump in last two frames
centroid_id5 = []
for f in range(611, 614):
    slc_id_5 = tracked_detections_all_frames[f]["ids"] == 5
    xyxy = tracked_detections_all_frames[f]["tracked_boxes"][slc_id_5, :]
    # Compute centroid
    xc = (xyxy[:, 0] + xyxy[:, 2]) / 2
    yc = (xyxy[:, 1] + xyxy[:, 3]) / 2
    centroid_id5.append(np.c_[xc, yc])  # concatenate along column axis


centroid_id5 = np.concatenate(centroid_id5)

# Compare
# why not exact same values?
print(centroid_id5)
print(ds.position.sel(individuals="id_5", time=slice(611, 613)).values)


# %%
