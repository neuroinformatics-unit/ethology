"""Add annotation_stats module for computing statistics.

This will be useful for understanding the dataset.

PURPOSE OF THIS MODULE
----------------------
Before training any model, it's important to understand the dataset.
This module provide simple functions to compute key statistics.

All functions work with standard ethology dataset as returned by
load_bboxes.from_files().
"""

import numpy as np
import pandas as pd
import xarray as xr


def dataset_summary(ds: xr.Dataset) -> dict:
    """Compute a summary of the dataset.

    Combine all helper functions into a single dictionary.

    KEYS
    ----
    - n_images: total number of images in the dataset
    - n_annotations: total number of valid annotations in the dataset
    - n_categories: total number of categories in the dataset
    - annotations_per_image: dict with mean, std, min and max per image
    - class_distribution: dict of category_name: count
    - bbox_size: dict with mean width, height and area

    Examples
    --------
    >>> from ethology.utils.annotation_stats import dataset_summary
    >>>from ethology.io.annotations.load_bboxes import from_files
    >>> ds = from_files("path/to/annotations_folder")
    >> summary = dataset_summary(ds)
    >>> print(summary["n_annotations"])

    """
    ann_per_img = annotations_per_image(ds)
    cls_dist = class_distribution(ds)
    bbox_stats = bbox_size_distribution(ds)

    # Handle empty annotations_per_image
    if ann_per_img.empty:
        ann_mean = ann_std = 0.0
        ann_min = ann_max = 0
    else:
        ann_mean = float(ann_per_img.mean())
        ann_std = float(ann_per_img.std())
        ann_min = int(ann_per_img.min())
        ann_max = int(ann_per_img.max())

    # Handle empty bbox_stats DataFrame
    # Ensures mean width, height, area are 0 if no boxes exist
    if bbox_stats.empty or bbox_stats["width"].empty:
        mean_width = mean_height = mean_area = 0.0
    else:
        mean_width = round(float(bbox_stats["width"].mean()), 2)
        mean_height = round(float(bbox_stats["height"].mean()), 2)
        mean_area = round(float(bbox_stats["area"].mean()), 2)

    return {
        "n_images": int(ds.sizes.get("image_id", 0)),
        "n_annotations": int(_count_valid(ds)),
        "n_categories": int(len(ds.attrs.get("map_category_to_str", {}))),
        "annotations_per_image": {
            "mean": ann_mean,
            "std": ann_std,
            "min": ann_min,
            "max": ann_max,
        },
        "class_distribution": cls_dist.to_dict(),
        "bbox_size": {
            "mean_width": mean_width,
            "mean_height": mean_height,
            "mean_area": mean_area,
        },
    }


def class_distribution(ds: xr.Dataset) -> pd.Series:
    """Count annotations per category across all images.

    Padding slots(category == -1) are excluded from the count.

    Returns
    -------
    pd.Series
        Counts indexed by category name (if available in attrs)
        or category ID.

    """
    # Into a 1D array by flatten()
    # shape (n_images, n_max_annot) → shape (n_images * n_max_annot,)
    cat_flat = ds["category"].values.flatten()
    # Remove padding slots
    valid_cats = cat_flat[cat_flat != -1]
    # Category ID
    counts = pd.Series(valid_cats, name="category_id").value_counts()
    # IDs <<>> Names
    cat_map = ds.attrs.get("map_category_to_str", {})
    if cat_map:
        counts.index = [
            cat_map.get(int(cat_id), str(cat_id)) for cat_id in counts.index
        ]

    counts.index.name = "category"
    counts.name = "count"

    return counts


def annotations_per_image(ds: xr.Dataset) -> pd.Series:
    """Count annotations in each image.

    Padding slots (category == -1) are excluded.

    Returns
    -------
    pd.Series
    Annotation counts indexed by image_id.

    """
    cat = ds["category"].values
    # axis=1 means sum across the id axis
    counts = (cat != -1).sum(axis=1)

    return pd.Series(
        counts, index=ds.coords["image_id"].values, name="annotation_count"
    )


def bbox_size_distribution(ds: xr.Dataset) -> pd.DataFrame:
    """Compute width, height, and area for each annotation.

    Returns
    -------
    pd.DataFrame
    One row per valid annotation with columns:
    width, height, area.

    """
    # Extract width and height
    widths = ds["shape"].sel(space="x").values
    heights = ds["shape"].sel(space="y").values
    cats = ds["category"].values

    rows = []
    n_images, n_max_annot = cats.shape

    for img_idx in range(n_images):
        for slot_idx in range(n_max_annot):
            # Skip padding slots
            if cats[img_idx, slot_idx] == -1:
                continue

            w = float(widths[img_idx, slot_idx])
            h = float(heights[img_idx, slot_idx])
            # Skip NaN values
            if np.isnan(w) or np.isnan(h):
                continue

            rows.append(
                {
                    "width": w,
                    "height": h,
                    "area": w * h,
                }
            )

    df = pd.DataFrame(rows)
    # Ensure consistent columns even if no valid annotations
    if df.empty:
        df = pd.DataFrame(columns=["width", "height", "area"])
    return df


def _count_valid(ds: xr.Dataset) -> int:
    # Count non-padding annotation slots
    return int((ds["category"].values != -1).sum())
