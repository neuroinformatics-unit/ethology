import numpy as np
from ethology.trackers.co_tracker import run_co_tracker
from ethology.visualisation.visualizer import Visualizer

# Provided input coordinates in [frame, y, x] format:
input_coords = np.array([
    [  0.        , 315.46836051, 255.54401554],
    [  0.        , 470.45975407, 382.35515573],
    [  0.        , 403.5316523 , 780.40123464],
    [  0.        , 230.92760038, 586.66199269]
])

# Since these coordinates are for frame 0, remove the first column and reorder to [x, y]:
# (Our model expects an array of shape (num_points, 2) in (x, y) order.)
# For testing purposes, you need a video file.
# Replace 'dummy_video.mp4' with the path to a valid video file.
video_path = "single-crab_MOCA-crab-1_video.mp4"

# Run the Co-Tracker backend with these query points.
# (This will load the video, preprocess frames, and run the model.)

res_video, pred_tracks, pred_visibility = run_co_tracker(video_path, input_coords)