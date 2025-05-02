# %%
import csv
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import xarray as xr
from boxmot import BotSort
from movement.io import load_bboxes, load_poses

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input data

# There is an ID swap in frames 1520, 2695, around 5411, 8234 and 8235
poses_file = Path("pose_data/SLEAP_two-mice_octagon.analysis.h5")

video_file = Path(
    "/home/sminano/swc/project_ethology/boxmot/pose_data/two-mice_octagon_video.mp4"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read pose dataset

ds = load_poses.from_sleap_file(poses_file)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Compute equivalentbounding boxes dataset

# Compute xmin, ymin, xmax, ymax of bounding boxes around each individual
# Ideally we can select the keypoints to use to compute the bounding boxes
xy_min = ds.position.min(dim="keypoints")
xy_max = ds.position.max(dim="keypoints")

# Compute centroid of each bounding box
xy_centroid = (xy_min + xy_max) / 2

# Compute width and height of each bounding box
width_height = xy_max - xy_min

ds_bboxes = load_bboxes.from_numpy(
    position_array=xy_centroid.values,
    shape_array=width_height.values,
    confidence_array=ds.confidence.mean("keypoints").values,
    individual_names=ds.individuals.values.tolist(),
    frame_array=ds.time.values.reshape(-1, 1),
    fps=None,
    source_software="SLEAP",
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Check in napari
# by write bounding boxes dataset as VIA-tracks file

# TODO: do this properly


def write_ds_bboxes_as_via_tracks(ds_bboxes: xr.Dataset, filename: str):
    # timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"pose_data/{filename}_bboxes_{timestamp}.csv"

    # Convert ds_bboxes to tracked_bboxes_dict format
    tracked_bboxes_dict = {}

    for t in ds_bboxes.time.values:
        # Get boxes at this timepoint
        boxes = []
        ids = []
        scores = []

        for ind in ds_bboxes.individuals.values:
            # Get centroid and shape
            centroid = ds_bboxes.position.sel(time=t, individuals=ind).values
            shape = ds_bboxes.shape.sel(time=t, individuals=ind).values

            # Skip if centroid is NaN
            if np.isnan(centroid).any():
                continue

            # Convert to xmin,ymin,xmax,ymax format
            xmin = centroid[0] - shape[0] / 2
            ymin = centroid[1] - shape[1] / 2
            xmax = centroid[0] + shape[0] / 2
            ymax = centroid[1] + shape[1] / 2

            boxes.append([xmin, ymin, xmax, ymax])
            ids.append(int(ind))  # Get ID number from individual name
            scores.append(1.0)  # Default confidence score

        # Add to dict
        tracked_bboxes_dict[int(t)] = {
            "tracked_boxes": np.array(boxes),
            "ids": np.array(ids),
            "scores": np.array(scores),
        }

    # Write to CSV file
    write_tracked_detections_to_csv(
        csv_file_path=filename,
        tracked_bboxes_dict=tracked_bboxes_dict,
    )


write_ds_bboxes_as_via_tracks(ds_bboxes, poses_file.stem)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run tracking on bounding boxes detections


# Initialize the tracker
tracker = BotSort(
    reid_weights=Path("osnet_x0_25_msmt17.pt"),  # Path to ReID model
    device="cpu",  # why not device? why is this in GPU if we then copy to CPU?
    half=False,
)

# Initialise dict to store tracked bboxes
tracked_detections_all_frames = {}

# Loop over frames
# frame_idx = 0
input_video_object = open_video(video_file)
total_n_frames = int(input_video_object.get(cv2.CAP_PROP_FRAME_COUNT))
t = 0
while input_video_object.isOpened():
    # Read frame
    ret, frame = input_video_object.read()

    # If ret is False, it means we have reached the end of the video
    if not ret:
        break

    # Get detections for this frame
    centroids = ds_bboxes.position.sel(time=t).values.T  # --> N X 2
    shapes = ds_bboxes.shape.sel(time=t).values.T  # --> N X 2
    confidences = ds_bboxes.confidence.sel(time=t).values  # --> N

    # Remove detections with NaN values
    valid_idx = ~np.isnan(centroids).any(axis=1)
    centroids = centroids[valid_idx, :]
    shapes = shapes[valid_idx, :]
    confidences = confidences[valid_idx]

    # Format detections as numpy array (N X (x, y, x, y, conf, cls))
    xmin = centroids[:, 0] - shapes[:, 0] / 2
    ymin = centroids[:, 1] - shapes[:, 1] / 2
    xmax = centroids[:, 0] + shapes[:, 0] / 2
    ymax = centroids[:, 1] + shapes[:, 1] / 2

    detections_array = np.column_stack(
        (xmin, ymin, xmax, ymax, confidences, np.ones_like(confidences))
    )

    # Update the tracker
    tracked_boxes_array = tracker.update(
        detections_array, frame
    )  # --> M X (x, y, x, y, id, conf, cls, ind)
    # ind is the index of the corresponding detection in the detections_array

    # Check if tracked_boxes_array is empty
    if tracked_boxes_array.shape[0] == 0:
        # No detections, skip this frame
        t += 1
        continue

    tracked_detections_all_frames[t] = {
        "tracked_boxes": tracked_boxes_array[:, :4],  # :-1],  # (x, y, x, y)
        "ids": tracked_boxes_array[
            :, 4
        ],  # -1],  # IDs are the last(5th) column
        "scores": tracked_boxes_array[:, 5],
    }

    # Update frame index
    t += 1

# Release video object
input_video_object.release()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Write tracked bboxes as VIA-tracks file
# to inspect in napari

# timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
filename = f"{poses_file.stem}_bboxes_boxMOT_{timestamp}.csv"

write_tracked_detections_to_csv(
    csv_file_path=filename,
    tracked_bboxes_dict=tracked_detections_all_frames,
)


# %%
# read as a movement dataset
# finds 12 individuals
# also sometimes the trajectory is ahead of the animal?
# maybe tweaking the parameters of the tracker?
ds_bboxes_tracked = load_bboxes.from_via_tracks_file(filename)
# %%
