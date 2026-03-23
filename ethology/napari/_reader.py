"""Napari reader plugin for ethology bounding box annotation files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

LayerData = tuple[Any, dict, str]


def napari_get_reader(path: str | list[str]):
    """Return a reader function if the path is a supported annotation file.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    callable or None
        A reader function if the path is a supported annotation file,
        otherwise None.

    """
    if isinstance(path, list):
        path = path[0]
    if not str(path).endswith(".json"):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if _is_coco_file(data) or _is_via_file(data):
            return _reader_function
    except Exception:
        return None
    return None


def _is_coco_file(data: dict) -> bool:
    """Return True if the dict matches the top-level COCO structure."""
    return all(k in data for k in ["images", "annotations", "categories"])


def _is_via_file(data: dict) -> bool:
    """Return True if the dict matches the top-level VIA structure."""
    return all(k in data for k in ["_via_img_metadata", "_via_attributes"])


def _detect_format(path: str | Path) -> str | None:
    """Detect the annotation format (COCO or VIA) of a JSON file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    str or None
        "COCO", "VIA", or None if the format cannot be determined.

    """
    try:
        with open(path) as f:
            data = json.load(f)
        if _is_coco_file(data):
            return "COCO"
        if _is_via_file(data):
            return "VIA"
    except Exception:  # pragma: no cover
        pass
    return None


def _reader_function(path: str | list[str]) -> list[LayerData]:
    """Read a bounding box annotation file as napari Shapes layer data.

    Parameters
    ----------
    path : str or list of str
        Path to a COCO or VIA annotation JSON file.

    Returns
    -------
    list of LayerData
        A list containing one tuple of (shapes_data, layer_kwargs, "shapes").
        Returns an empty list if the file cannot be read.

    """
    from ethology.io.annotations.load_bboxes import from_files

    if isinstance(path, list):
        path = path[0]

    fmt = _detect_format(path)
    if fmt is None:
        return []

    ds = from_files(path, format=fmt)
    return _dataset_to_napari_shapes(ds)


def _dataset_to_napari_shapes(ds: Any) -> list[LayerData]:
    """Convert an ethology bounding box dataset to napari Shapes layer data.

    Each non-NaN bounding box in the dataset is converted to a napari
    rectangle defined by its four corner points in (row, col) = (y, x) order.

    Parameters
    ----------
    ds : xarray.Dataset
        A valid ethology bounding box annotations dataset.

    Returns
    -------
    list of LayerData
        A list with a single LayerData tuple for a napari Shapes layer.

    """
    shapes: list[np.ndarray] = []
    shape_types: list[str] = []
    image_ids: list[int] = []
    annotation_ids: list[int] = []
    category_ids: list[int] = []

    has_category = "category" in ds.data_vars
    space_vals = list(ds.space.values)
    x_idx = space_vals.index("x")
    y_idx = space_vals.index("y")

    position_vals = ds.position.values  # (n_images, 2, n_annots)
    shape_vals = ds.shape.values  # (n_images, 2, n_annots)
    category_vals = ds.category.values if has_category else None

    for img_i in range(ds.sizes["image_id"]):
        image_id = int(ds.image_id.values[img_i])
        for ann_i in range(ds.sizes["id"]):
            x_center = position_vals[img_i, x_idx, ann_i]
            y_center = position_vals[img_i, y_idx, ann_i]
            if np.isnan(x_center) or np.isnan(y_center):
                continue

            width = shape_vals[img_i, x_idx, ann_i]
            height = shape_vals[img_i, y_idx, ann_i]
            x_min = x_center - width / 2.0
            y_min = y_center - height / 2.0
            x_max = x_center + width / 2.0
            y_max = y_center + height / 2.0

            # napari rectangle: four corners in (y, x) order
            rect = np.array(
                [
                    [y_min, x_min],
                    [y_min, x_max],
                    [y_max, x_max],
                    [y_max, x_min],
                ],
                dtype=np.float64,
            )
            shapes.append(rect)
            shape_types.append("rectangle")
            image_ids.append(image_id)
            annotation_ids.append(ann_i)
            cat = (
                int(category_vals[img_i, ann_i])
                if has_category and category_vals is not None
                else -1
            )
            category_ids.append(cat)

    properties = {
        "image_id": np.array(image_ids, dtype=int),
        "annotation_id": np.array(annotation_ids, dtype=int),
        "category_id": np.array(category_ids, dtype=int),
    }
    layer_kwargs: dict[str, Any] = {
        "name": "bounding_boxes",
        "shape_type": shape_types,
        "properties": properties,
        "edge_width": 2,
        "face_color": "transparent",
        "metadata": {
            "annotation_format": ds.attrs.get("annotation_format", "COCO"),
            "annotation_files": ds.attrs.get("annotation_files", ""),
            "images_directories": ds.attrs.get("images_directories", None),
            "map_category_to_str": ds.attrs.get("map_category_to_str", {}),
            "map_image_id_to_filename": ds.attrs.get(
                "map_image_id_to_filename", {}
            ),
        },
    }
    return [(shapes, layer_kwargs, "shapes")]
