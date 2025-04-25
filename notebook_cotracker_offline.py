# %%
# Imports
# import sleap_io as sio
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

# %matplotlib widget

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Data paths
video_path = "/home/sminano/swc/project_ethology/tap_models_crabs/input/04.09.2023-04-Right_RE_test.mp4"

ground_truth_data = Path(
    "/home/sminano/swc/project_ethology/tap_models_crabs/input/04.09.2023-04-Right_RE_test_corrected_ST_SM_20241029_113207.csv"
)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Query points from gt

ds_gt = load_bboxes.from_file(
    file_path=ground_truth_data,
    source_software="VIA-tracks",
    use_frame_numbers_from_file=False,
)

# Prepare query points array
# it has frame as first column
queries_array = np.vstack(
    [
        np.hstack(
            [
                f * np.ones((ds_gt.sizes["individuals"], 1)),  # frame column
                ds_gt.position.sel(time=f).values.T,  # x, y columns
            ]
        )
        for f in range(ds_gt.sizes["time"])
    ]
)

# Remove rows with nans in position
queries_array = queries_array[~np.any(np.isnan(queries_array), axis=1), :]

# # Select frames
# list_frames = list(range(ds_gt.sizes["time"]))
# frames_to_select = np.array(list_frames)[
#     ::step_between_query_frames
# ]  # every second frame
# queries_sel = queries_array[[col in frames_to_select for col in queries_array[:, 0]], :]

# print(np.unique(queries_sel[:, 0]))


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Read video
# TODO: is it faster with sleap_io?
video_full = read_video_from_path(video_path)
print(type(video_full))
print(video_full.shape)  # (614, 2160, 4096, 3)


# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Make it a torch tensor with dimensions in expected order
video_full = torch.from_numpy(video_full).permute(0, 3, 1, 2)[None]

print(video_full.shape)  # (1, 614, 3, 2160, 4096)
print(video_full.device)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Downsample frames
# out_frame_size = [216, 410] # 108, 205
# video = F.interpolate(video[0], out_frame_size, mode="bilinear")[None]

print(video_full.shape)  # (1, 614, 3, 2160, 4096)
print(video_full.device)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Select first part of the video only  (to fit in GPU)
# video = video[:, : video.shape[1] // 8]
chunk_start = 0
video = video_full[:, chunk_start : chunk_start + 75, :, :, :]  # 75 frames
print(video.shape)  # (1, 307, 3, 2160, 4096)

# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# Convert to float and place video on device
# Why do we need .float conversion?
# chatgpt: Mathematical operations like convolutions, normalizations, or matrix mults expect float32 or float16


device = "cuda"
# video = video.float().to(device)
# video = video.half().to(device) # Use half precision for memory efficiency
# TODO: Make sure your video is normalized properly (video / 255.0) before converting to half()
video = video.to(torch.float).to(device)  # torch.float16

# %%
# Check gpu memory usage
# print(torch.cuda.memory_summary())
# %%
# Define query points
queries = torch.tensor(
    [
        [0.0, 1070.1, 1697.1],
        # if downsampled: [0.0, 97.09, 177.34],  # point tracked from the first frame
        [0.0, 980.7, 1762.2],
        # if downsampled: [0.0, 106.20, 170.33],
        # [113.0, 1961.00, 1665.00]
        # [10.0, 600.0, 500.0],  # frame number 10
        # [20.0, 750.0, 600.0],  # ...
        # [30.0, 900.0, 200.0],
    ]
)

# # Select all points at the first frame of the chunk
# queries = queries_array[queries_array[:, 0] == chunk_start, :]  
# queries = queries[:1, :]
# queries = torch.tensor(queries)

# Place query tensor on GPU
queries = queries.to(torch.float).to(device)  # .half().to(device) torch.float16

# %%
# Visualize query points over frame

# Create a list of frame numbers corresponding to each point
frame_numbers = queries[:, 0].int().unique().tolist()

for frame_number in frame_numbers:
    # get the query points for the current frame
    queries_one_frame = queries[queries[:, 0] == frame_number]

    fig, ax = plt.subplots(1, 1)
    # plot frame
    ax.imshow(
        video_full[frame_number, :, :]
    )  # B T C H W -> H W C
    # plot query points
    ax.scatter(
        x=queries_one_frame[:, 1].cpu(), y=queries_one_frame[:, 2].cpu(), s=5, c="red"
    )

    ax.set_title("Frame {}".format(frame_number))
    ax.set_xlim(0, video.shape[4])
    ax.set_ylim(0, video.shape[3])
    ax.invert_yaxis()


# %%
# Get Offline CoTracker model
model = torch.hub.load("facebookresearch/co-tracker", "cotracker3_offline")

# Use the model in half precision and move it to the GPU
# Note: this is for memory usage
model = model.to(device)  # .half().to(device) # .to(torch.float16).to(device)


# %%
# all_half = all(p.dtype == torch.float16 for p in model.parameters())
# print("All parameters are float16:", all_half)

# for name, param in model.named_parameters():
#     # print(f"{name}: {param.dtype}")
#     if param.dtype == torch.float32:
#         param.data = param.data.to(torch.float16)
#         print("PATATA")

# for name, buffer in model.named_buffers():
#     print(f"{name}: {buffer.dtype}")


# %%
# Run CoTracker
pred_tracks, pred_visibility = model(
    video, queries=queries[None], backward_tracking=True
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
# Save as a movement dataset

# Assuming 1 query per individual

# Prepare position array
# (n_frames, n_space, n_keypoints, n_individuals)
position_array = np.empty(
    ds_gt.position.shape[:2] + (1,) + (ds_gt.position.shape[-1],)
    # add keypoint dimension
)
position_array.fill(np.nan)
position_array[150 : 150 + 75, :, :, :] = (
    pred_tracks.permute(1, -1, 0, -2).cpu().numpy()
)  

ds = load_poses.from_numpy(
    position_array=position_array,
    individual_names=[f"ind_{i}" for i in range(pred_tracks.shape[2])],
    keypoint_names=["centroid"],
    source_software="CoTracker3",
)


# Export to read in napari
ds.attrs["source_file"] = ""

# get string timestamp of  today in yyyymmdd_hhmmss
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

save_poses.to_sleap_analysis_file(
    ds,
    f"output/cotracker_offline_output_{timestamp}.h5",
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
