"""Module for exporting manually labelled bounding boxes."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytz

from ethology.annotations.io.load_bboxes import STANDARD_BBOXES_DF_INDEX
from ethology.annotations.validators import ValidCOCO

# Mapping of dataframe columns to COCO keys
STANDARD_BBOXES_DF_COLUMNS_TO_COCO = {
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


def _validate_df_bboxes(df: pd.DataFrame):
    """Check if the input dataframe is a valid bounding boxes dataframe."""
    # Check type
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"Expected a pandas DataFrame, but got {type(df)}.")

    # Check index name is as expected
    if df.index.name != STANDARD_BBOXES_DF_INDEX:
        raise ValueError(
            f"Expected index name to be '{STANDARD_BBOXES_DF_INDEX}', "
            f"but got '{df.index.name}'."
        )

    # Check bboxes coordinates exist as df columns
    if not all(x in df.columns for x in ["x_min", "y_min", "width", "height"]):
        raise ValueError(
            "Required bounding box coordinates "
            "'x_min', 'y_min', 'width', 'height', are not present in "
            "the dataframe."
        )


def _fill_in_COCO_required_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return the bboxes input dataframe with any COCO required data added."""
    # Add annotation_id as column
    df["annotation_id"] = df.index

    # Add COCO required data
    if "category" not in df.columns:
        df["category"] = ""  # if not defined: set as empty string

    if "category_id" not in df.columns or df["category_id"].dtype != int:
        df["category_id"] = df["category"].factorize(sort=True)[0]

    if "area" not in df.columns:
        df["area"] = df["width"] * df["height"]

    if "iscrowd" not in df.columns:
        df["iscrowd"] = 0  # if not defined: assume default value

    if "segmentation" not in df.columns:
        # If not defined: assume default value for iscrowd=0
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

    return df


def _create_COCO_dict(df: pd.DataFrame) -> dict:
    """Extract COCO dictionary from a bounding boxes dataframe."""
    COCO_dict: dict[str, Any] = {}
    for sections in ["images", "categories", "annotations"]:
        # Extract required columns
        df_section = df[
            list(STANDARD_BBOXES_DF_COLUMNS_TO_COCO[sections].keys())
        ].copy()

        # Rename columns to COCO standard
        df_section = df_section.rename(
            columns=STANDARD_BBOXES_DF_COLUMNS_TO_COCO[sections]
        )

        # Extract rows as lists of dictionaries
        if sections == "annotations":
            row_dicts = df_section.to_dict(orient="records")
        else:
            row_dicts = df_section.drop_duplicates().to_dict(orient="records")

        # Append to COCO_dict
        COCO_dict[sections] = row_dicts

    # Add info section to COCO_dict
    COCO_dict["info"] = {
        "date_created": datetime.now(pytz.utc).strftime(
            "%a %b %d %Y %H:%M:%S GMT%z"
        ),
        "description": "Bounding boxes annotations exported from ethology",
        "url": "https://github.com/neuroinformatics-unit/ethology",
    }

    return COCO_dict


def to_COCO_file(df: pd.DataFrame, output_filepath: str | Path):
    """Write bounding boxes annotations to a COCO JSON file.

    Parameters
    ----------
    df : pd.DataFrame
        Bounding boxes annotations dataframe.
    output_filepath : str or Path
        Output file path.

    Returns
    -------
    output_filepath : str
        Output file path.

    """
    # Validate input dataframe
    _validate_df_bboxes(df)

    # Sort, drop duplicate annotations and reindex
    df = df.sort_values(by=["image_filename"])
    df = df.drop_duplicates(ignore_index=True)

    # Fill in COCO required data
    df = _fill_in_COCO_required_data(df)

    # Create COCO dictionary
    COCO_dict = _create_COCO_dict(df)

    # Write to JSON file
    with open(output_filepath, "w") as f:
        json.dump(COCO_dict, f, sort_keys=True, indent=2)

    # Check if output file is a valid COCO for ethology
    ValidCOCO(output_filepath)

    return output_filepath
