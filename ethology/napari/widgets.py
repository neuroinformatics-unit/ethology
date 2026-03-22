"""Napari widget for reviewing and correcting bounding box annotations."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari


def make_correction_widget(napari_viewer: "napari.Viewer"):
    """Return a widget for loading and saving bounding box annotations.

    The widget exposes two panels:

    - **Load annotations**: reads a COCO or VIA annotation file and adds the
      bounding boxes as a napari Shapes layer for interactive correction.
    - **Save corrected annotations**: writes the (corrected) active Shapes
      layer back to a COCO JSON file.

    Parameters
    ----------
    napari_viewer : napari.Viewer
        The running napari viewer instance (injected by napari).

    Returns
    -------
    magicgui.widgets.Container
        A container holding the two sub-widgets.
    """
    from magicgui import magicgui
    from magicgui.widgets import Container

    @magicgui(
        call_button="Load annotations",
        annotation_file={"label": "Annotation file", "mode": "r"},
        format={"label": "Format", "choices": ["COCO", "VIA"]},
    )
    def load_widget(annotation_file: Path = Path("."), format: str = "COCO"):
        """Load bounding box annotations as a napari Shapes layer."""
        from ethology.io.annotations.load_bboxes import from_files
        from ethology.napari._reader import _dataset_to_napari_shapes

        try:
            ds = from_files(str(annotation_file), format=format)
            layer_data_list = _dataset_to_napari_shapes(ds)
            if layer_data_list:
                shapes, kwargs, _ = layer_data_list[0]
                napari_viewer.add_shapes(shapes, **kwargs)
        except Exception as exc:
            import warnings

            warnings.warn(
                f"Could not load annotations from {annotation_file}: {exc}",
                stacklevel=2,
            )

    @magicgui(
        call_button="Save corrected annotations",
        output_file={"label": "Output COCO file", "mode": "w"},
    )
    def save_widget(output_file: Path = Path("corrected.json")):
        """Save the active Shapes layer to a COCO annotation file."""
        from ethology.napari._writer import write_shapes

        shapes_layers = [
            layer
            for layer in napari_viewer.layers
            if hasattr(layer, "shape_type")
        ]
        if not shapes_layers:
            import warnings

            warnings.warn(
                "No Shapes layer found in the napari viewer. "
                "Load annotations first.",
                stacklevel=2,
            )
            return

        layer = shapes_layers[-1]
        meta = {
            "shape_type": list(layer.shape_type),
            "properties": {k: v for k, v in layer.properties.items()},
            "metadata": layer.metadata,
        }
        try:
            written = write_shapes(str(output_file), layer.data, meta)
            print(f"Saved corrected annotations to: {written[0]}")
        except Exception as exc:
            import warnings

            warnings.warn(
                f"Could not save annotations to {output_file}: {exc}",
                stacklevel=2,
            )

    return Container(widgets=[load_widget, save_widget])