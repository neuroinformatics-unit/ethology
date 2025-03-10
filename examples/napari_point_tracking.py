from typing import List

import napari
import pandas as pd
from dask_image.imread import imread
from magicgui.widgets import ComboBox, Container, PushButton

from ethology.trackers import BaseTracker

COLOR_CYCLE = [
    '#1f77b4',
    '#ff7f0e',
    '#2ca02c',
    '#d62728',
    '#9467bd',
    '#8c564b',
    '#e377c2',
    '#7f7f7f',
    '#bcbd22',
    '#17becf'
]


def create_label_menu(points_layer, labels):
    """Create a label menu widget that can be added to the napari viewer dock

    Parameters
    ----------
    points_layer : napari.layers.Points
        a napari points layer
    labels : List[str]
        list of the labels for each keypoint to be annotated (e.g., the body parts to be labeled).

    Returns
    -------
    label_menu : Container
        the magicgui Container with our dropdown menu widget
    """
    # Create the label selection menu
    label_menu = ComboBox(label='feature_label', choices=labels)
    label_widget = Container(widgets=[label_menu])


    def update_label_menu(event):
        """Update the label menu when the point selection changes"""
        new_label = str(points_layer.feature_defaults['label'][0])
        if new_label != label_menu.value:
            label_menu.value = new_label

    points_layer.events.feature_defaults.connect(update_label_menu)

    def label_changed(selected_label):
        """Update the Points layer when the label menu selection changes"""
        feature_defaults = points_layer.feature_defaults
        feature_defaults['label'] = selected_label
        points_layer.feature_defaults = feature_defaults
        points_layer.refresh_colors()

    label_menu.changed.connect(label_changed)

    return label_widget

def create_cotrack_runner(vid_path, points_layer, labels):
    """Create a label menu widget that can be added to the napari viewer dock

    Parameters
    ----------
    points_layer : napari.layers.Points
        a napari points layer
    labels : List[str]
        list of the labels for each keypoint to be annotated (e.g., the body parts to be labeled).

    Returns
    -------
    label_menu : Container
        the magicgui Container with our dropdown menu widget
    """
    # Create the label selection menu
    # label_menu = ComboBox(label='feature_label', choices=labels)
    button = PushButton(text="Run tracker on selected points.")

    cotrack_widget = Container(widgets=[button])

    @button.clicked.connect
    def on_button_click():
        tracker = BaseTracker(tracker_type="cotracker")
        # Output is in shape [n_batch=1, n_frames, n_keypoints, n_space=2]
        trajectories = tracker.track(vid_path, points_layer.data, save_results=True)

    return cotrack_widget

def point_annotator(
        vid_path: str,
        labels: List[str],
):
    """Create a GUI for annotating points in a series of images.

    Parameters
    ----------
    vid_path : str
        glob-like string for the video to be labeled.
    labels : List[str]
        list of the labels for each keypoint to be annotated (e.g., the body parts to be labeled).
    """
    stack = imread(vid_path)

    viewer = napari.view_image(stack)
    points_layer = viewer.add_points(
        ndim=3,
        features=pd.DataFrame({'label': pd.Categorical([], categories=labels)}),
        border_color='label',
        border_color_cycle=COLOR_CYCLE,
        symbol='o',
        face_color='transparent',
        border_width=0.5,  # fraction of point size
        size=12,
    )
    points_layer.border_color_mode = 'cycle'

    # add the label menu widget to the viewer
    label_widget = create_label_menu(points_layer, labels)
    cotrack_widget = create_cotrack_runner(vid_path, points_layer, labels)


    viewer.window.add_dock_widget(label_widget)
    viewer.window.add_dock_widget(cotrack_widget)

    @viewer.bind_key('.')
    def next_label(event=None):
        """Keybinding to advance to the next label with wraparound"""
        feature_defaults = points_layer.feature_defaults
        default_label = feature_defaults['label'][0]
        ind = list(labels).index(default_label)
        new_ind = (ind + 1) % len(labels)
        new_label = labels[new_ind]
        feature_defaults['label'] = new_label
        points_layer.feature_defaults = feature_defaults
        points_layer.refresh_colors()

    def next_on_click(layer, event):
        """Mouse click binding to advance the label when a point is added"""
        if layer.mode == 'add':
            # By default, napari selects the point that was just added.
            # Disable that behavior, as the highlight gets in the way
            # and also causes next_label to change the color of the
            # point that was just added.
            layer.selected_data = set()
            next_label()

    points_layer.mode = 'add'
    points_layer.mouse_drag_callbacks.append(next_on_click)

    @viewer.bind_key(',')
    def prev_label(event):
        """Keybinding to decrement to the previous label with wraparound"""
        feature_defaults = points_layer.feature_defaults
        default_label = feature_defaults['label'][0]
        ind = list(labels).index(default_label)
        n_labels = len(labels)
        new_ind = ((ind - 1) + n_labels) % n_labels
        feature_defaults['label'] = labels[new_ind]
        points_layer.feature_defaults = feature_defaults
        points_layer.refresh_colors()

    napari.run()


video_source = "apple.mov"
point_annotator(video_source, ['stem', 'core'])