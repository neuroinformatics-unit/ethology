# %%
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2 as transforms
import xarray as xr
from boxmot import BoostTrack
from movement.io import save_poses
from tqdm import tqdm

from ethology.detectors.ensembles import combine_detections_across_models_wbf
from ethology.detectors.inference import (
    concat_detections_ds,
    detections_dict_as_ds,
)
from ethology.detectors.load import load_fasterrcnn_resnet50_fpn_v2
from ethology.detectors.utils import (
    add_bboxes_min_max_corners,
    detections_ds_as_x1y1_x2y2,
    detections_ds_to_movement_ds,
    tracks_x1y1_x2y2_as_ds,
)
from ethology.mlflow import (
    read_cli_args_from_mlflow_params,
    read_config_from_mlflow_params,
    read_mlflow_params,
)

# Set xarray options
xr.set_options(display_expand_attrs=False)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data

video_path = Path(
    "/home/sminano/swc/project_ethology/04.09.2023-04-Right_RE_test.mp4"
)

experiment_ID = "617393114420881798"
ml_runs_experiment_dir = (
    Path("/home/sminano/swc/project_crabs/ml-runs") / experiment_ID
)

# I pick seed 42 for each set of models
models_dict = {
    "above_0th": ml_runs_experiment_dir / "f348d9d196934073bece1b877cbc4d38",
    "above_1st": ml_runs_experiment_dir / "879d2f77e2b24adcb06b87d2fede6a04",
    "above_5th": ml_runs_experiment_dir / "75583ec227e3444ab692b99c64795325",
    "above_10th": ml_runs_experiment_dir / "4acc37206b1e4f679d535c837bee2c2f",
    "above_25th": ml_runs_experiment_dir / "fdcf88fcbcc84fbeb94b45ca6b6f8914",
    "above_50th": ml_runs_experiment_dir / "daa05ded0ea047388c9134bf044061c5",
}

output_dir = Path("/home/sminano/swc/project_ethology/ensemble_tracking_output")

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Create output directory
output_dir.mkdir(parents=True, exist_ok=True)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Set default device: CUDA if available, otherwise mps, otherwise CPU
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

print(f"Using device: {device}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Helper functions
def open_video(video_path: str | Path) -> cv2.VideoCapture:
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
    with open(csv_file_path, "w") as csv_file:
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

    return csv_file_path


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define list of models in ensemble

list_models = []
list_config = []
list_cli_args = []
for model_key in models_dict:
    # Retrieve model config and CLI args from mlflow
    trained_model_path = str(
        models_dict[model_key] / "checkpoints" / "last.ckpt"
    )

    mlflow_params = read_mlflow_params(trained_model_path)
    config = read_config_from_mlflow_params(mlflow_params)
    cli_args = read_cli_args_from_mlflow_params(mlflow_params)

    print(
        f"Run name: {mlflow_params['run_name']}, trained on "
        f"{Path(cli_args['dataset_dirs'][0]).name}, "
        f"{Path(cli_args['annotation_files'][0]).name}"
    )
    # ------------------------------------
    # Load model
    model = load_fasterrcnn_resnet50_fpn_v2(
        trained_model_path,
        num_classes=config["num_classes"],
        device=device,  # device
    )
    model.eval()
    list_models.append(model)
    list_config.append(config)
    list_cli_args.append(cli_args)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check that all models have the same dataset config
ref_config = list_config[0]
for key in ["train_fraction", "val_over_test_fraction"]:
    assert all(config[key] == ref_config[key] for config in list_config)

ref_cli_args = list_cli_args[0]
assert all(
    cli_args["seed_n"] == ref_cli_args["seed_n"] for cli_args in list_cli_args
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Define transforms for inference
inference_transforms = transforms.Compose(
    [
        transforms.ToImage(),
        transforms.ToDtype(torch.float32, scale=True),
    ]
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Helper function: run detector on video


def run_detector_on_video(
    model: torch.nn.Module,
    video_path: str | Path,
    device: torch.device,
    inference_transforms: transforms.Compose,
) -> xr.Dataset:
    """Run detection on a video."""

    # Ensure model is in evaluation mode
    model.eval()

    # Loop over frames
    list_detections_ds = []
    list_image_ids = []
    frame_idx = 0
    input_video_object = open_video(video_path)
    # total_n_frames = int(input_video_object.get(cv2.CAP_PROP_FRAME_COUNT))
    while input_video_object.isOpened():
        # Read frame
        ret, frame = input_video_object.read()
        if not ret:
            break  # end of video or error

        # Place image tensor on device and add batch dimension
        image_tensor = inference_transforms(frame).to(device)[None]

        # Run detection
        with torch.no_grad():
            # use [0] to select the one image in the batch
            # Returns: dictionary with data of the predicted bounding boxes.
            # The keys are: "boxes", "scores", and "labels". The labels
            # refer to the class of the object detected, and not its ID.
            detections = model(image_tensor)

        # Format as xarray dataset
        # [0] to select single batch dimension
        detections_ds = detections_dict_as_ds(detections[0])

        # Append to list
        list_detections_ds.append(detections_ds)
        list_image_ids.append(frame_idx)

        # Update frame index
        frame_idx += 1

    # Concatenate all detections datasets along image_id dimension
    detections_dataset = concat_detections_ds(
        list_detections_ds,
        pd.Index(list_image_ids, name="image_id"),
    )

    # Add image_width and image_height as attributes
    # (we assume all images in the dataset have the same width and height
    # as the last image)
    detections_dataset.attrs["image_width"] = image_tensor.shape[-1]  # columns
    detections_dataset.attrs["image_height"] = image_tensor.shape[-2]  # rows

    return detections_dataset


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run ensemble on video
# can I make it faster?
# can I vectorize this? (pytorch forum question)
list_detections_ds = []
for model in tqdm(list_models):
    model.to(device)

    detections_ds = run_detector_on_video(
        model=model,
        video_path=video_path,
        device=device,
        inference_transforms=inference_transforms,
    )
    # detections_ds = add_bboxes_min_max_corners(detections_ds)
    list_detections_ds.append(detections_ds)

# Concatenate detections across models
all_models_detections_ds = concat_detections_ds(
    list_detections_ds,
    pd.Index(range(len(list_detections_ds)), name="model"),
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Fuse detections across models

all_models_detections_ds = add_bboxes_min_max_corners(all_models_detections_ds)

confidence_th_post_fusion = 0.4
fused_detections_ds = combine_detections_across_models_wbf(
    all_models_detections_ds,
    kwargs_wbf={
        "iou_thr_ensemble": 0.5,
        "skip_box_thr": 0.0001,
        "max_n_detections": 300,
        "confidence_th_post_fusion": confidence_th_post_fusion,
    },
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Format detections as a movement dataset

# add id coordinate (FIX this)
fused_detections_ds = fused_detections_ds.assign_coords(
    id=np.arange(fused_detections_ds.sizes["id"])
)

# format as movement dataset
fused_detections_as_movement_ds = detections_ds_to_movement_ds(
    fused_detections_ds, type="poses"
)

# save as sleap analysis file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_poses.to_sleap_analysis_file(
    fused_detections_as_movement_ds,
    output_dir / f"detections_ensemble_{video_path.stem}_{timestamp}.h5",
)



# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Track detections using boxmot

# Initialize the tracker
# tracker = BotSort(
#     reid_weights=Path("osnet_x0_25_msmt17.pt"),  # Path to ReID model
#     device=device,  # "0" # why not device? why is this in GPU if we then copy to CPU?
#     half=False,
# )

tracker = BoostTrack(
    reid_weights=Path("osnet_x0_25_msmt17.pt"),
    device=device,
    half=False,
    max_age=1000, # frames
    min_hits=1,
    det_thresh=0,  # already filtered by confidence_th_post_fusion
    iou_threshold=0.1,  # for association
    aspect_ratio_thresh=1000,
    min_box_area=0, # no minimum box area
)


# %%

# TODO: vectorize with apply_ufunc?
list_tracked_ds = []
input_video_object = open_video(video_path)
for image_id in np.sort(fused_detections_ds.image_id.values):
    # Convert detections to numpy arrays
    detections_one_img_ds = fused_detections_ds.sel(image_id=image_id)
    x1y1_x2y2_array, scores_array, labels_array = detections_ds_as_x1y1_x2y2(
        detections_one_img_ds
    )

    # Get frame from video
    ret, frame = input_video_object.read()
    if not ret:
        break

    # Update the tracker
    #   INPUT:  M X (x, y, x, y, conf, cls)
    #   OUTPUT: M X (x, y, x, y, id, conf, cls, ind)
    # ind is the index of the corresponding detection in the detections_array
    tracked_boxes_array = tracker.update(
        np.c_[x1y1_x2y2_array, scores_array, labels_array],
        frame,
    )

    # # Can I do away with reordering the predictions
    # ind = tracked_boxes_array[:, -1].astype(int)
    # # select detections
    # tracked_ds = detections_one_img_ds.sel(id=ind)
    # # reorder ids per frame
    # detections_one_img_ds = detections_one_img_ds.reindex({"id": ind})
    # # reset index to 0
    # detections_one_img_ds = detections_one_img_ds.assign_coords(
    #     {"id": range(len(ind))}
    # )

    tracked_ds = tracks_x1y1_x2y2_as_ds(
        tracked_boxes_array[:, :4],  # centroid x1y1x2y2
        tracked_boxes_array[:, 5],  # confidence
        tracked_boxes_array[:, 6].astype(int),  # label
        tracked_boxes_array[:, 4].astype(int),  # id
    )

    # add image_id coordinate
    tracked_ds = tracked_ds.assign_coords(image_id=image_id)

    list_tracked_ds.append(tracked_ds)

# Concatenate all tracked detections datasets along image_id dimension
tracked_ds_all_frames = concat_detections_ds(
    list_tracked_ds, pd.Index(range(len(list_tracked_ds)), name="image_id")
)
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Format tracked detections dataset as a movement dataset

# reindex id coordinate to start from 0
tracked_ds_all_frames = tracked_ds_all_frames.assign_coords(
    id=np.arange(tracked_ds_all_frames.sizes["id"])
)

# format as movement dataset
tracks_as_movement_ds = detections_ds_to_movement_ds(
    tracked_ds_all_frames, type="poses"
)


# save as sleap analysis file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_poses.to_sleap_analysis_file(
    tracks_as_movement_ds,
    output_dir / f"tracks_ensemble_{video_path.stem}_{timestamp}.h5",
)


# %%
