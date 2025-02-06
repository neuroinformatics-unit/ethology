"""Module for exporting manually labelled bounding boxes."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytz

from ethology.annotations.validators import ValidCOCO


def df_bboxes_to_COCO_file(df: pd.DataFrame, output_filepath: str | Path):
    """Write bounding boxes annotations to a COCO JSON file."""
    # Map dataframe columns to COCO fields for renaming
    map_columns_to_COCO_keys = {
        "images": {
            "image_id": "id",
            "image_filename": "file_name",
            "image_width": "width",
            "image_height": "height",
        },
        "categories": {
            "category_id": "id",
            "category": "name",
            "supercategory": "supercategory",
        },
        "annotations": {
            "annotation_id": "id",
            "area": "area",
            "bbox": "bbox",
            "image_id": "image_id",
            "category_id": "category_id",
            "iscrowd": "iscrowd",
            "segmentation": "segmentation",
        },
    }

    #########
    # Add missing data if required
    # add annotation_id as column
    if "annotation_id" not in df.columns:
        df["annotation_id"] = df.index

    # add category_id if missing
    if "category_id" not in df.columns:
        # VIA exports COCO files with specified category_id;
        # If not defined: we assume category_id is assigned starting
        # from 0 and considering alphabetical order of categories
        df["category_id"] = df["category"].factorize(sort=True)[0]

    if "area" not in df.columns:
        df["area"] = df["width"] * df["height"]

    if "iscrowd" not in df.columns:
        df["iscrowd"] = 0  # if not defined: assume default value

    if "segmentation" not in df.columns:
        # If not defined: assume default segmentation for iscrowd=0
        # Default is a polygon defined by the 4 corners of the bounding box

        # Compute 4 corners of the bounding box
        top_left_corner = df[["x_min", "y_min"]].to_numpy()
        delta_xy = df[["width", "height"]].to_numpy()
        delta_x_only = np.vstack([df["width"], np.zeros_like(df["height"])]).T
        delta_y_only = np.vstack([np.zeros_like(df["width"]), df["height"]]).T

        # Combine all xy coordinates of corners into one column
        df["segmentation"] = np.hstack(
            [
                top_left_corner,
                top_left_corner + delta_x_only,  # top right corner
                top_left_corner + delta_xy,  # bottom right corner
                top_left_corner + delta_y_only,  # bottom left corner
            ]
        ).tolist()

        # Wrap in a list of lists, to match VIA format
        df["segmentation"] = df["segmentation"].apply(lambda x: [x])

    if "bbox" not in df.columns:
        df["bbox"] = (
            df[["x_min", "y_min", "width", "height"]].to_numpy().tolist()
        )

    #########
    COCO_dict: dict[str, Any] = {}
    for sections in ["images", "categories", "annotations"]:
        # Extract required columns
        df_section = df[list(map_columns_to_COCO_keys[sections].keys())].copy()

        # Rename columns
        df_section = df_section.rename(
            columns=map_columns_to_COCO_keys[sections]
        )

        # Extract rows as lists of dictionaries
        if sections == "annotations":
            list_dict = df_section[
                list(map_columns_to_COCO_keys[sections].values())
            ].to_dict(orient="records")
        else:
            list_dict = (
                df_section[list(map_columns_to_COCO_keys[sections].values())]
                .drop_duplicates()  # unique images and categories only
                .to_dict(orient="records")
            )

        # Append to COCO_dict
        COCO_dict[sections] = list_dict

    #################
    # Add info section with metadata?
    # contributor, date_created, description, url, version, year
    # Format the timestamp
    formatted_timestamp = datetime.now(pytz.utc).strftime(
        "%a %b %d %Y %H:%M:%S GMT%z"
    )
    COCO_dict["info"] = {
        "date_created": formatted_timestamp,
        "description": "Bounding boxes annotations exported from ethology",
        "url": "https://github.com/neuroinformatics-unit/ethology",
    }

    ################
    # Write to JSON file
    with open(output_filepath, "w") as f:
        json.dump(COCO_dict, f, sort_keys=True, indent=4)

    # Check if valid COCO
    ValidCOCO(output_filepath)

    return output_filepath
