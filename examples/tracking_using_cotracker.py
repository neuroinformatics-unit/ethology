"""Example of tracking using the cotracker."""

from ethology.trackers import BaseTracker

video_source = "../ethology/trackers/cotracker/assets/apple.mp4"

query_points = [
    [0, 400, 350],  # frame_number, x, y
    [0, 600, 500],
    [0, 750, 600],
    [0, 900, 200],
]

tracker = BaseTracker(tracker_type="cotracker")

path = tracker.track(video_source, query_points, save_results=True)
