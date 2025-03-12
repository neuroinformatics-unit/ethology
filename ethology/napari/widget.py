"""Widget module for Napari integration with ethology.

This file defines a Napari dock widget for browsing a video file,
running any-point tracking, and displaying results.
"""

import napari
from napari.layers import Points
from napari_video.napari_video import VideoReaderNP
from qtpy.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from ethology.trackers.co_tracker import run_co_tracker


class AnyPointTrackingWidget(QWidget):
    """A Napari dock widget for any-point tracking."""

    def __init__(self, napari_viewer):
        """Initialize the widget.

        Parameters
        ----------
        napari_viewer : napari.Viewer
            The napari viewer instance.

        """
        super().__init__()
        self.viewer = napari_viewer
        self.video_path = ""
        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
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
        """Open a file dialog to select a video and load it in the viewer."""
        video_file, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video File",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv)",
        )
        if video_file:
            self.video_path = video_file
            self.file_line_edit.setText(video_file)
            print(f"Selected video: {video_file}")

            try:
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
        """Run the any-point tracker and display the results."""
        # Check if a "Query Points" layer exists in the viewer
        if "Points" not in self.viewer.layers:
            print(
                "No 'Points' layer found. Please add one "
                "and place points on the image."
            )
            return

        query_points_layer = self.viewer.layers["Points"]
        query_points = (
            query_points_layer.data
        )  # Expected to be an Nx2 array of coordinates

        if query_points.size == 0:
            print("The 'Points' layer is empty. Please add some points.")
            return

        if not self.video_path:
            print("Please select a video file before running tracking.")
            return

        print(self.video_path)
        print(query_points)
        # Run the tracker function (replace with your actual model call)
        res_video, pred_tracks, pred_visibility = run_co_tracker(
            self.video_path, query_points
        )

        # Visualize the results.
        # Here, for demonstration, we add a points layer
        # with the first frame's points.
        if "Result Video" in self.viewer.layers:
            self.viewer.layers.remove("Result Video")

        # Here, we assume res_video is in a format acceptable
        # by napari.add_image (e.g. a numpy array of shape
        # [T, H, W] or [T, H, W, C]).
        self.viewer.add_image(res_video, name="Result Video")
        print("Result video layer added.")

        print("Tracking complete. Displaying results for all frames.")


def on_new_layer(event):
    """Handle insertion of a new layer in the viewer.

    Parameters
    ----------
    event : napari.utils.events.Event
        The event containing the new layer.

    """
    new_layer = event.value
    if isinstance(new_layer, Points) and new_layer.name == "Points":
        print(f"A new Points layer named '{new_layer.name}' was added!")
        new_layer.events.data.connect(on_points_changed)


def on_points_changed(event):
    """Handle changes to the data in a Points layer.

    Parameters
    ----------
    event : napari.utils.events.Event
        The event containing updated points data.

    """
    points_coords = event.value
    print("Updated points coordinates:", points_coords)
    global global_widget
    if global_widget is not None:
        global_widget.query_points = points_coords


if __name__ == "__main__":
    """Run the widget standalone for testing purposes."""
    viewer = napari.Viewer()
    global_widget = None
    viewer.layers.events.inserted.connect(on_new_layer)
    widget = AnyPointTrackingWidget(viewer)
    global_widget = widget
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()
