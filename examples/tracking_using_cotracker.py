"""Example of tracking using the cotracker."""

from movement.io import load_poses

from ethology.trackers import BaseTracker

video_source = "../ethology/trackers/cotracker/assets/apple.mp4"

query_points = [
    [0, 400, 350],  # frame_number, x, y
    [0, 600, 500],
    [0, 750, 600],
    [0, 900, 200],
]

tracker = BaseTracker(tracker_type="cotracker")

# Output is in shape [n_batch=1, n_frames, n_keypoints, n_space=2]
trajectories = tracker.track(video_source, query_points, save_results=True)

# Movement needs arrays shaped (n_frames, n_space, n_keypoints, n_individuals)
# Assume n_batch = 1 = n_individuals
ds = load_poses.from_numpy(
    trajectories.numpy().transpose(1, 3, 2, 0), source_software="cotracker"
)

print(ds)
