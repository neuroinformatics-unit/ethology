"""Napari writer plugin for ethology bounding box annotation files."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr


def write_shapes(path: str, data: list, meta: dict) -> list[str]:
    """Write a napari Shapes layer of bounding boxes to a COCO JSON file.

    Only shapes of type ``rectangle`` are written. Non-rectangle shapes
    are silently skipped.

    Parameters
    ----------
    path : str
        Destination path for the output COCO JSON file.
    data : list of numpy.ndarray
        List of shape arrays, each with shape ``(N, 2)`` in (y, x) order,
        as returned by ``napari.layers.Shapes.data``.
    meta : dict
        Layer keyword arguments as returned by napari. Must contain:
        ``shape_type`` (list of str), ``properties`` (dict), and
        ``metadata`` (dict) as produced by :func:`_reader_function`.

    Returns
    -------
    list of str
        List containing the path to the written file.
    """
    from ethology.io.annotations.save_bboxes import to_COCO_file

    shape_types: list[str] = meta.get("shape_type", [])
    properties: dict = meta.get("properties", {})
    layer_metadata: dict = meta.get("metadata", {})

    ds = _shapes_to_dataset(data, shape_types, properties, layer_metadata)
    to_COCO_file(ds, output_filepath=path)
    return [path]


def _shapes_to_dataset(
    shapes: list[np.ndarray],
    shape_types: list[str],
    properties: dict,
    layer_metadata: dict,
) -> xr.Dataset:
    """Convert napari Shapes layer data to an ethology xarray dataset.

    Parameters
    ----------
    shapes : list of numpy.ndarray
        List of shape arrays, each ``(N, 2)`` in (y, x) order.
    shape_types : list of str
        Shape type for each entry in ``shapes``.
    properties : dict
        Layer properties (image_id, annotation_id, category_id arrays).
    layer_metadata : dict
        Layer metadata from the ethology reader plugin.

    Returns
    -------
    xarray.Dataset
        A valid ethology bounding box annotations dataset ready for
        :func:`~ethology.io.annotations.save_bboxes.to_COCO_file`.

    Raises
    ------
    ValueError
        If no rectangle shapes are found in the input data.
    """
    raw_image_ids = np.asarray(
        properties.get("image_id", np.zeros(len(shapes), dtype=int)),
        dtype=int,
    )
    raw_category_ids = np.asarray(
        properties.get("category_id", np.full(len(shapes), -1, dtype=int)),
        dtype=int,
    )

    rows: list[dict[str, Any]] = []
    for i, (shape, stype) in enumerate(zip(shapes, shape_types)):
        if stype != "rectangle":
            continue
        ys = shape[:, 0]
        xs = shape[:, 1]
        x_min, x_max = float(xs.min()), float(xs.max())
        y_min, y_max = float(ys.min()), float(ys.max())
        width = x_max - x_min
        height = y_max - y_min
        rows.append(
            {
                "image_id": int(raw_image_ids[i]),
                "x_center": x_min + width / 2.0,
                "y_center": y_min + height / 2.0,
                "width": width,
                "height": height,
                "category_id": int(raw_category_ids[i]),
            }
        )

    if not rows:
        raise ValueError(
            "No rectangle shapes found in the Shapes layer. "
            "Only rectangles can be exported as bounding box annotations."
        )

    # Group rows by image_id preserving order
    unique_image_ids: list[int] = list(
        dict.fromkeys(r["image_id"] for r in rows)
    )
    rows_by_image: dict[int, list[dict]] = {
        img_id: [] for img_id in unique_image_ids
    }
    for row in rows:
        rows_by_image[row["image_id"]].append(row)

    n_images = len(unique_image_ids)
    max_annots = max(len(v) for v in rows_by_image.values())

    position_data = np.full((n_images, 2, max_annots), np.nan, dtype=float)
    shape_data = np.full((n_images, 2, max_annots), np.nan, dtype=float)
    category_data = np.full((n_images, max_annots), -1, dtype=int)

    for img_i, img_id in enumerate(unique_image_ids):
        for ann_i, row in enumerate(rows_by_image[img_id]):
            position_data[img_i, 0, ann_i] = row["x_center"]  # x
            position_data[img_i, 1, ann_i] = row["y_center"]  # y
            shape_data[img_i, 0, ann_i] = row["width"]
            shape_data[img_i, 1, ann_i] = row["height"]
            category_data[img_i, ann_i] = row["category_id"]

    ds = xr.Dataset(
        data_vars={
            "position": (["image_id", "space", "id"], position_data),
            "shape": (["image_id", "space", "id"], shape_data),
            "category": (["image_id", "id"], category_data),
        },
        coords={
            "image_id": unique_image_ids,
            "space": ["x", "y"],
            "id": range(max_annots),
        },
    )

    # Restore or build the required dataset attributes
    map_image_id_to_filename: dict = layer_metadata.get(
        "map_image_id_to_filename"
    ) or {img_id: f"image_{img_id}.jpg" for img_id in unique_image_ids}

    unique_cat_ids = set(int(c) for c in category_data.flatten() if c != -1)
    map_category_to_str: dict = layer_metadata.get(
        "map_category_to_str"
    ) or {cat_id: f"category_{cat_id}" for cat_id in sorted(unique_cat_ids)}

    ds.attrs = {
        "annotation_files": layer_metadata.get("annotation_files", ""),
        "annotation_format": layer_metadata.get("annotation_format", "COCO"),
        "images_directories": layer_metadata.get("images_directories", None),
        "map_category_to_str": map_category_to_str,
        "map_image_id_to_filename": map_image_id_to_filename,
    }
    return ds