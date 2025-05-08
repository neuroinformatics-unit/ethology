"""Sliding-window approach for online tracking with CoTracker3.

Todo:
- more query points?
- longer window better?
- overlapping window better?

"""

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
import os
from datetime import datetime
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
from movement.io import load_bboxes, load_poses, save_poses

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Input video
video_path = Path(
    "/home/sminano/swc/project_ethology/tap_models_crabs/input/04.09.2023-04-Right_RE_test.mp4"
)

window_half_length = 40  # in frames, overlapping,
step_between_query_frames = 100


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Query points

ground_truth_data = Path(
    "/home/sminano/swc/project_ethology/tap_models_crabs/input/04.09.2023-04-Right_RE_test_corrected_ST_SM_20241029_113207.csv"
)

ds_gt = load_bboxes.from_file(
    file_path=ground_truth_data,
    source_software="VIA-tracks",
    use_frame_numbers_from_file=False,
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Select query points
# ------------------
# Select one individual only
ds_gt_one = ds_gt.isel(individuals=[13 - 1])  # [3-1,13-1,35-1,57-1])

# Select all individuals
# ds_gt_one = ds_gt

print(ds_gt_one)

# Select frames
list_frames = list(range(ds_gt_one.sizes["time"]))
frames_to_select = np.array(list_frames)[
    ::step_between_query_frames
]  # every N frame
print(frames_to_select)
# --------------------

# Prepare query points array
# it has frame as first column
queries_array = np.vstack(
    [
        np.hstack(
            [
                f
                * np.ones((ds_gt_one.sizes["individuals"], 1)),  # frame column
                ds_gt_one.position.sel(time=f).values.T,  # x, y columns
            ]
        )
        for f in range(ds_gt_one.sizes["time"])
    ]
)

# Remove rows with nans in position
queries_array = queries_array[~np.any(np.isnan(queries_array), axis=1), :]

# Selected queries
queries_sel = queries_array[
    [col in frames_to_select for col in queries_array[:, 0]], :
]
print(np.unique(queries_sel[:, 0]))

# %%
# Convert to torch tensor
queries = torch.tensor(queries_sel)
queries = queries.to(torch.float).to(DEFAULT_DEVICE)


# %%%%%%%%%%%%%%%%%%%
# Load online model

model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_online")

# Set window length
model.model.window_len = window_half_length * 2  # in frames

# Set step
model.step = window_half_length
print(model.step)  # window is of width model.step * 2

# Move to GPU
model = model.to(DEFAULT_DEVICE)

# model = CoTrackerOnlinePredictor(
#     # checkpoint=None,
#     checkpoint=(
#       "/home/sminano/swc/project_ethology/tap_models_crabs/"
#       "scaled_online.pth"
#     ),
#     window_len=2 * window_half_length,  # in frames
#     v2=False,
# )


# %%%%%%%%%%%%%%%
# Process chunk function


def _process_step(window_frames, is_first_step, queries):
    # Get a chunk of the video
    video_chunk = (
        torch.tensor(
            np.stack(window_frames[-model.step * 2 :]), device=DEFAULT_DEVICE
        )
        .float()
        .permute(0, 3, 1, 2)[None]
    )  # (1, T, 3, H, W)

    # Process the video chunk with the model
    return model(
        video_chunk,
        is_first_step=is_first_step,
        queries=queries[None],
    )


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Process video in non-overlapping chunks
window_frames: list[np.ndarray] = []

# Iterating over video frames, processing one window at a time:
is_first_step = True
video_iterator = iio.imiter(str(video_path), plugin="FFMPEG")
for i, frame in enumerate(video_iterator):
    # Process a video chunk (non-overlapping right?)
    if i % model.step == 0 and i != 0:
        pred_tracks, pred_visibility = _process_step(
            window_frames,
            is_first_step,
            queries=queries,
        )
        is_first_step = False

    # append frame to window_frames
    window_frames.append(frame)


# Processing the final video frames
# (in case video length is not a multiple of model.step)
pred_tracks, pred_visibility = _process_step(
    window_frames[-(i % model.step) - model.step - 1 :],
    is_first_step,
    queries=queries,
    # grid_query_frame=i
)

print("Tracks are computed")

# %%
print(pred_tracks)
print(pred_tracks.shape)  # (1, T, N, 2)
print(
    "pred_tracks in MB: "
    f"{(pred_tracks.element_size() * pred_tracks.nelement()) / 1e6}"
)

print(pred_visibility.shape)  # (1, T, N, 1)
print(
    "pred_visibility in MB: "
    f"{(pred_visibility.element_size() * pred_visibility.nelement()) / 1e6}"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save as a movement dataset
# (n_frames, n_space, n_keypoints, n_individuals)

# assuming 1 query per individual in frame 0
position_array = (
    pred_tracks.permute(1, -1, 0, -2).cpu().numpy()
)  # (T, 2, 1, Nqueries)
visibility_array = pred_visibility.cpu().numpy()[0]  # (T, Nqueries)

# set to nan if non visible
# (improve this)
for i in range(visibility_array.shape[1]):
    position_array[~visibility_array[:, i], :, :, i] = np.nan

# -----------------------------
# # get each track from its query point
# position_array_fix = np.vstack(
#     [
#         position_array[
#             frames_to_select[i]:(frames_to_select[i+1]
#               if i<queries.shape[0]-1 else None), :, i
#         ]
#         for i in range(queries.shape[0])
#     ]
# )
# position_array_fix = position_array_fix.T[None,None].T
# --------------------------------------------

ds = load_poses.from_numpy(
    position_array=position_array,  # position_array_fix,
    individual_names=[f"ind_{i}" for i in range(position_array.shape[-1])],
    keypoint_names=["centroid"],
    source_software="CoTracker3",
)


# Export to read in napari
ds.attrs["source_file"] = ""

# get string timestamp of  today in yyyymmdd_hhmmss
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

save_poses.to_sleap_analysis_file(
    ds,
    f"../tap_models_crabs/output/cotracker_output_{timestamp}.h5",
)


# %%
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
print(f"Reserved:  {torch.cuda.memory_reserved() / 1024**2:.2f} MB")


# %%
def model_gpu_mem_MB(model):
    """Calculate the GPU memory usage of a PyTorch model."""
    total = 0
    for param in model.parameters():
        if param.is_cuda:
            total += param.element_size() * param.nelement()
    for buffer in model.buffers():
        if buffer.is_cuda:
            total += buffer.element_size() * buffer.nelement()
    return total / 1024 / 1024  # Convert bytes to MB


print(f"Model uses approximately {model_gpu_mem_MB(model):.2f} MB on GPU")

# %%
# Remove model from GPU?
# del model

# %%

# Save a video with predicted tracks

# vis = Visualizer(save_dir="output", pad_value=120, linewidth=3)
# vis.visualize(video, pred_tracks, pred_visibility,
# query_frame=grid_query_frame)


# # Generate timestamp of today in format YYYYMMDD_HHMMSS
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# # Place video on gpu
# video = read_video_from_path(video_path)
# video = torch.from_numpy(video).permute(0, 3, 1, 2)[None]
# print((video.element_size() * video.nelement()) / 1e9)  # in GB

# video = video.to(DEFAULT_DEVICE)  # (1, T, 3, H, W) ---> OOM


# # %%
# vis = Visualizer(
#     save_dir="./output",
#     linewidth=1,
#     mode="cool",
#     tracks_leave_trace=-1,
#     fps=10,
# )

# vis.visualize(
#     video,
#     pred_tracks,  # .to('cpu'),
#     pred_visibility,
#     query_frame=grid_query_frame,
#     filename=f"queries_{timestamp}",
# )

# %%
