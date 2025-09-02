"""Module for exporting manually labelled bounding boxes."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytz
import xarray as xr

from ethology.io.annotations.validate import (
    STANDARD_BBOXES_DF_COLUMNS_TO_COCO,
    STANDARD_BBOXES_DF_INDEX,
    ValidCOCO,
    validate_df_bboxes,
)


def _annotations_ds_to_df(ds: xr.Dataset) -> pd.DataFrame:
    """Convert annotations xarray dataset to a dataframe.

    Parameters
    ----------
    ds : xr.Dataset
        Bounding boxes annotations xarray dataset.

    Returns
    -------
    df : pd.DataFrame
        Bounding boxes annotations dataframe.

    """
    # Create dataframe from xarray dataset
    df_raw = ds.to_dataframe(dim_order=["image_id", "id", "space"])
    df_raw = df_raw.reset_index()

    # Remove rows where position or shape data is nan
    # (where at least one of the specified columns contains a NaN value.)
    df_raw = df_raw.dropna(subset=["position", "shape"])

    # Pivot the dataframe to get position_x, position_y, shape_x, shape_y
    index_cols = ["image_id", "id", "category"]
    df_raw = df_raw.pivot_table(
        index=index_cols,
        columns="space",
        values=["position", "shape"],
    ).reset_index()

    # Flatten the columns
    df_raw.columns = [
        "_".join(col).strip() if col[1] != "" else col[0]
        for col in df_raw.columns.values
    ]

    # Rename "id" to "id_per_frame" and "category" to "category_id"
    # (Note that category in dataset is an integer)
    df_raw.rename(columns={"id": "id_per_frame"}, inplace=True)
    df_raw.rename(columns={"category": "category_id"}, inplace=True)

    # Compute x_min, y_min, width, height for each annotation
    df_raw["x_min"] = df_raw["position_x"] - df_raw["shape_x"] / 2
    df_raw["y_min"] = df_raw["position_y"] - df_raw["shape_y"] / 2
    df_raw["width"] = df_raw["shape_x"]
    df_raw["height"] = df_raw["shape_y"]

    # Compute "image_filename" from "image_id"
    map_image_id_to_filename = ds.attrs["map_image_id_to_filename"]
    df_raw["image_filename"] = df_raw["image_id"].map(map_image_id_to_filename)

    # Compute "category" as string from "category_id"
    map_category_to_str = ds.attrs["map_category_to_str"]
    df_raw["category"] = df_raw["category_id"].map(map_category_to_str)

    # Set index name to STANDARD_BBOXES_DF_INDEX
    df_raw.index.name = STANDARD_BBOXES_DF_INDEX

    # Select columns to keep
    cols_to_select = [
        "image_id",
        "image_filename",
        "x_min",
        "y_min",
        "width",
        "height",
        "category",
        "category_id",
        "id_per_frame",
    ]
    df_output = df_raw[cols_to_select]

    return df_output


def _fill_in_COCO_required_data(df: pd.DataFrame) -> pd.DataFrame:
    """Return the bboxes input dataframe with any COCO required data added.

    Parameters
    ----------
    df : pd.DataFrame
        Bounding boxes dataframe for "annotations" or "predictions".

    Returns
    -------
    df : pd.DataFrame
        Bounding boxes dataframe with COCO required data added.

    """
    # Add "annotation_id" as column
    df["annotation_id"] = df.index

    # Add COCO required data
    # if not defined: set as empty string
    if "category" not in df.columns:
        df["category"] = ""

    # if "supercategory" not defined: set as empty string
    if "supercategory" not in df.columns:
        df["supercategory"] = ""

    # if "category_id" not defined or not an integer: set as
    # 1-based integer factorized category
    if "category_id" not in df.columns or df["category_id"].dtype != int:
        try:
            df["category_id"] = df["category_id"].astype(int)
        except ValueError:
            df["category_id"] = df["category"].factorize(sort=True)[0] + 1

    # if "area" not defined: set as width * height
    if "area" not in df.columns:
        df["area"] = df["width"] * df["height"]

    # if "iscrowd" not defined: set as 0 (default value)
    if "iscrowd" not in df.columns:
        df["iscrowd"] = 0

    # if "segmentation" not defined: set as a polygon defined by the
    # 4 corners of the bounding box
    if "segmentation" not in df.columns:
        top_left_corner = df[["x_min", "y_min"]].to_numpy()
        delta_xy = df[["width", "height"]].to_numpy()
        delta_x_only = np.vstack([df["width"], np.zeros_like(df["height"])]).T
        delta_y_only = np.vstack([np.zeros_like(df["width"]), df["height"]]).T

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

    # if "bbox" not defined: set as x_min, y_min, width, height
    if "bbox" not in df.columns:
        df["bbox"] = (
            df[["x_min", "y_min", "width", "height"]].to_numpy().tolist()
        )

    # if "image_width" not defined: set as 0
    if "image_width" not in df.columns:
        df["image_width"] = 0

    # if "image_height" not defined: set as 0
    if "image_height" not in df.columns:
        df["image_height"] = 0

    return df


def _create_COCO_dict(df: pd.DataFrame) -> dict:
    """Extract COCO dictionary from a bounding boxes dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Bounding boxes dataframe for "annotations" or "predictions".

    Returns
    -------
    COCO_dict : dict
        COCO dictionary.

    """
    COCO_dict: dict[str, Any] = {}
    for sections in ["images", "categories", "annotations"]:
        # Extract required columns
        list_required_columns = list(
            STANDARD_BBOXES_DF_COLUMNS_TO_COCO[sections].keys()
        )
        if "confidence" in list_required_columns:
            list_required_columns.remove("confidence")

        df_section = df[list_required_columns].copy()

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


def to_COCO_file(ds: xr.Dataset, output_filepath: str | Path):
    """Write bounding boxes annotations dataset to a COCO JSON file.

    Parameters
    ----------
    ds : xr.Dataset
        Bounding boxes annotations xarray dataset.
    output_filepath : str or Path
        Output file path.

    Returns
    -------
    output_filepath : str
        Output file path.

    """
    # Compute dataframe from xarray dataset
    df = _annotations_ds_to_df(ds)

    # Validate dataframe
    validate_df_bboxes(df)

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

    # Check output file is a valid COCO file for ethology
    ValidCOCO(output_filepath)

    return output_filepath
