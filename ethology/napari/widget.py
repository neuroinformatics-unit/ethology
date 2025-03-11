# run_any_point_tracking.py

from qtpy.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QLabel, QFileDialog
)
import napari
import numpy as np
from ethology.trackers.co_tracker import run_co_tracker


class AnyPointTrackingWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.video_path = ""
        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()

        # --- Video File Selection ---
        file_layout = QHBoxLayout()
        self.file_label = QLabel("Video File:")
        self.file_line_edit = QLineEdit()
        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.browse_video)

        file_layout.addWidget(self.file_label)
        file_layout.addWidget(self.file_line_edit)
        file_layout.addWidget(self.browse_button)
        layout.addLayout(file_layout)

        # --- Run Tracking Button ---
        self.run_button = QPushButton("Run Any-Point Tracking")
        self.run_button.clicked.connect(self.run_tracking)
        layout.addWidget(self.run_button)

        self.setLayout(layout)

    def browse_video(self):
        video_file, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", "", "Video Files (*.mp4 *.avi *.mov *.mkv)"
        )
        if video_file:
            self.video_path = video_file
            self.file_line_edit.setText(video_file)
            print(f"Selected video: {video_file}")

            try:
                from napari_video.napari_video import VideoReaderNP
                # Create a VideoReaderNP instance
                vr = VideoReaderNP(video_file)
                
                # Remove any existing 'Video' layer
                if "Video" in self.viewer.layers:
                    self.viewer.layers.remove("Video")
                    
                # Add the video as an image layer to the viewer.
                self.viewer.add_image(vr, name="Video")
                print("Video loaded successfully.")
            except Exception as e:
                print("Failed to load video using napari-video:", e)
        

    def run_tracking(self):
        # Check if a "Query Points" layer exists in the viewer
        if "Points" not in self.viewer.layers:
            print("No 'Points' layer found. Please add one and place points on the image.")
            return

        query_points_layer = self.viewer.layers["Points"]
        query_points = query_points_layer.data  # Expected to be an Nx2 array of coordinates

        if query_points.size == 0:
            print("The 'Points' layer is empty. Please add some points.")
            return

        if not self.video_path:
            print("Please select a video file before running tracking.")
            return

        print(self.video_path)
        print(query_points)
        # Run the tracker function (replace with your actual model call)
        res_video, pred_tracks, pred_visibility = run_co_tracker(self.video_path, query_points)
        
        # Visualize the results.
        # Here, for demonstration, we add a points layer with the first frame's points.
        # Option 2: Display the processed result video.
        if "Result Video" in self.viewer.layers:
            self.viewer.layers.remove("Result Video")
        
        # Here, we assume res_video is in a format acceptable by napari.add_image (e.g. a numpy array of shape [T, H, W] or [T, H, W, C]).
        self.viewer.add_image(res_video, name="Result Video")
        print("Result video layer added.")

        print("Tracking complete. Displaying results for all frames.")

if __name__ == '__main__':
    # Create a Napari viewer instance.
    viewer = napari.Viewer()

    global_widget = None

    from napari.layers import Points

    def on_new_layer(event):
        new_layer = event.value
        # Check if a new Points layer named "Query Points" is added.
        if isinstance(new_layer, Points) and new_layer.name == "Points":
            print(f"A new Points layer named '{new_layer.name}' was added!")
            # Connect to its data changes.
            new_layer.events.data.connect(on_points_changed)

    def on_points_changed(event):
        # event.value contains the updated coordinates (a NumPy array).
        points_coords = event.value
        print("Updated points coordinates:", points_coords)
        # Update the widget's stored query points so tracking uses the latest coordinates.
        if global_widget is not None:
            global_widget.query_points = points_coords

    # Create an instance of your widget and add it as a dock widget.
    widget = AnyPointTrackingWidget(viewer)
    viewer.window.add_dock_widget(widget, area='right')

    # Start Napari's event loop.
    napari.run()
