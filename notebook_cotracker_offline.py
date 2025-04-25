# %%
# Imports
import os
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from cotracker.utils.visualizer import read_video_from_path
from movement.io import load_bboxes, load_poses, save_poses

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


DEFAULT_DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

%matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Data paths
video_path = "/home/sminano/swc/project_ethology/tap_models_crabs/input/04.09.2023-04-Right_RE_test.mp4"

ground_truth_data = Path(
    "/home/sminano/swc/project_ethology/tap_models_crabs/input/04.09.2023-04-Right_RE_test_corrected_ST_SM_20241029_113207.csv"
)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Parmeters

# query points
step_between_query_frames = 5
individuals_gt_ids = [57]

# downsample video
scale_factor = 0.25

# clip video
chunk_start = 0
chunk_width = 75


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Select query points

ds_gt = load_bboxes.from_file(
    file_path=ground_truth_data,
    source_software="VIA-tracks",
    use_frame_numbers_from_file=False,
)


# ------------------
# Select individuals to use as query points
if len(individuals_gt_ids) == 0:
    ds_gt_one = ds_gt
else:
    ds_gt_one = ds_gt.isel(individuals=[i - 1 for i in individuals_gt_ids])

print(ds_gt_one)

# Select frames
list_frames = list(range(ds_gt_one.sizes["time"]))
frames_to_select = np.array(list_frames)[
    chunk_start:chunk_start + chunk_width:step_between_query_frames
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

# Filter selected query points
queries_sel = queries_array[
    [col in frames_to_select for col in queries_array[:, 0]], :
]

print(queries_sel.shape)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Downsample queries by the same scale factor as the video
queries_downsampled = queries_sel * scale_factor
queries_downsampled[:, 0] = queries_sel[:, 0]
print(queries_downsampled.shape)  # torch.Size([1, 614, 2])
print(queries_downsampled)

# convert to torch tensor and place on device
queries_downsampled = torch.tensor(queries_downsampled).to(torch.float).to(
    DEFAULT_DEVICE
)  # .half().to(device) torch.float16


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read video
# TODO: is it faster with sleap_io? yes! but then converting to torch is very slow
# %time video_full = read_video_from_path(video_path)  # Wall time: 13.4 s
# %time video_full = sio.load_video(video_path)  # Wall time: 27.4 ms
# %time video_full = np.array(sio.load_video(video_path))

video_full = read_video_from_path(video_path)
print(type(video_full))
print(video_full.shape)  # (614, 2160, 4096, 3)

# as torch tensor
video_full = torch.from_numpy(video_full).permute(0, 3, 1, 2)[None]

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Downsample video
video_downsampled = F.interpolate(
    video_full[0], scale_factor=scale_factor, mode="bilinear"
)[None]

print(video_downsampled.shape)  # torch.Size([1, 614, 3, 540, 1024])
print(video_downsampled.device)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Select first part of the video only  (to fit in GPU)
# video = video[:, : video.shape[1] // 8]
video_downsampled_chunk = video_downsampled[
    :, chunk_start : chunk_start + chunk_width, :, :, :
]  # 75 frames
print(video_downsampled_chunk.shape)  # torch.Size([1, 75, 3, 540, 1024])

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Convert to float and place video on device
# Why do we need .float conversion?
# chatgpt: Mathematical operations like convolutions, normalizations, or matrix mults expect float32 or float16


device = "cuda"
# video = video.float().to(device)
# video = video.half().to(device) # Use half precision for memory efficiency
# TODO: Make sure your video is normalized properly (video / 255.0) before converting to half()
video_downsampled_chunk = video_downsampled_chunk.to(torch.float).to(
    device
)  # torch.float16


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Visualize query points over frames

# Create a list of frame numbers corresponding to each point
frame_numbers = queries_downsampled[:, 0].int().unique().tolist()

for frame_number in frame_numbers:
    if frame_number in list(range(video_downsampled_chunk.shape[1])):
        # get the query points for the current frame
        queries_one_frame = queries_downsampled[
            queries_downsampled[:, 0] == frame_number
        ]

        fig, ax = plt.subplots(1, 1)
        # plot frame
        ax.imshow(
            video_downsampled_chunk.permute(0, 1, -2, -1, -3)[
                0, frame_number, :, :, :
            ]
            .cpu()
            .numpy()
            .astype(np.int32)
        )  # B T C H W -> H W C
        # plot query points
        ax.scatter(
            x=queries_one_frame[:, 1].cpu(),
            y=queries_one_frame[:, 2].cpu(),
            s=5,
            c="red",
        )

        ax.set_title("Frame {}".format(frame_number))
        ax.set_xlim(0, video_downsampled_chunk.shape[4])
        ax.set_ylim(0, video_downsampled_chunk.shape[3])
        ax.invert_yaxis()


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Get Offline CoTracker model
model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

# Use the model in half precision and move it to the GPU
# Note: this is for memory usage
model = model.to(device)  # .half().to(device) # .to(torch.float16).to(device)

print(model.model.window_len)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# all_half = all(p.dtype == torch.float16 for p in model.parameters())
# print("All parameters are float16:", all_half)

# for name, param in model.named_parameters():
#     # print(f"{name}: {param.dtype}")
#     if param.dtype == torch.float32:
#         param.data = param.data.to(torch.float16)
#         print("PATATA")

# for name, buffer in model.named_buffers():
#     print(f"{name}: {buffer.dtype}")


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Run CoTracker
pred_tracks, pred_visibility = model(
    video_downsampled_chunk, 
    queries=queries_downsampled[None], 
    backward_tracking=True,
)  # B T N 2,  B T N 1


# from torch.cuda.amp import autocast
# model.eval()
# with torch.no_grad(), torch.autocast(device_type="cuda"):
#     pred_tracks, pred_visibility = model(
#         video, queries=queries[None], #backward_tracking=True
#     )  # B T N 2,  B T N 1

# %%
# TODO: Can I upsample the results to the original video res?
print(pred_tracks.shape)  # (1, 307, 2, 2) --> Batch, Time, N of points, 2 (x,y)
print(pred_visibility.shape)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Upsample results to the original video resolution

pred_tracks_upsampled = pred_tracks*1 / scale_factor

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Save as a movement dataset
# (n_frames, n_space, n_keypoints, n_individuals)

# assuming 1 query is 1 individual
position_array = (
    pred_tracks_upsampled.permute(1, -1, 0, -2).cpu().numpy()
)  # (T, 2, 1, Nqueries)
visibility_array = pred_visibility.cpu().numpy()[0]  # (T, Nqueries)

# set position to nan if non visible
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
    f"../tap_models_crabs/output/cotracker_offline_output_{timestamp}.h5",
)


# %%
# # Visualize results

# vis = Visualizer(
#     save_dir="./output",
#     linewidth=1,
#     mode="cool",
#     tracks_leave_trace=-1,
#     fps=10,
# )

# # Generate timestamp of today in format YYYYMMDD_HHMMSS
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# # Save video with predictions
# vis.visualize(video, pred_tracks, pred_visibility, filename=f"queries_{timestamp}")

# %%
