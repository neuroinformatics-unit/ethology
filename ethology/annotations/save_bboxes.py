"""Module for exporting manually labelled bounding boxes."""

import json
from pathlib import Path

import pandas as pd

from ethology.annotations.validators import ValidCOCO


def df_bboxes_to_COCO_file(df: pd.DataFrame, output_filepath: str | Path):
    """Write bounding boxes annotations to a COCO JSON file."""
    # Map dataframe columns to COCO fields for renaming
    # (if a column is not defined, it is left as is)
    map_df_cols_to_COCO = {
        "images": {
            "image_id": "id",
            "image_filename": "file_name",
            "image_width": "width",
            "image_height": "height",
        },
        "categories": {
            "category_id": "id",
            "category": "name",
        },
        "annotations": {"annotation_id": "id"},
    }

    required_COCO_keys = {
        "images": ["id", "file_name", "width", "height"],
        "categories": ["id", "name", "supercategory"],
        "annotations": [
            "area",
            "bbox",
            "id",
            "image_id",
            "category_id",
            "iscrowd",
            "segmentation",
        ],
    }

    #########
    # Prepare image columns
    df = df.rename(columns=map_df_cols_to_COCO["images"])

    #########
    # Prepare category columns
    # add category_id if missing
    if "category_id" not in df.columns:
        df.loc[:, "category_id"] = df["category"].factorize()
    # rename
    df = df.rename(columns=map_df_cols_to_COCO["categories"])

    # Prepare annotations columns
    # add area, iscrowd, segmentation, bboxes data if missing
    df["area"] = df["width"] * df["height"]
    if "iscrowd" not in df.columns:
        df["iscrowd"] = 0  # if not defined: assume default value

    if "segmentation" not in df.columns:
        # if not defined: assume default value (polygon) for iscrowd=0
        df["segmentation_x0_y0"] = df[["x_min", "y_min"]]
        df["segmentation_x1_y1"] = df[["x_min", "y_min"]] + (df[["width"]], 0)
        df["segmentation_x2_y2"] = df[["x_min", "y_min"]] + (
            df[["width"]],
            df[["height"]],
        )
        df["segmentation_x3_y3"] = df[["x_min", "y_min"]] + (0, df[["height"]])

        # combine
        df["segmentation"] = df[
            [f"segmentation_x{i}_y{i}" for i in range(4)]
        ].values.tolist()

    if "bbox" not in df.columns:
        df["bbox"] = df[["x_min", "y_min", "width", "height"]].values.tolist()

    # rename
    df = df.rename(columns=map_df_cols_to_COCO["annotations"])

    # Extract list of images with required COCO keys
    list_images = (
        df[required_COCO_keys["images"]]
        .drop_duplicates()  # unique images only
        .to_dict(orient="records")
    )

    # Get list of categories
    list_categories = (
        df[required_COCO_keys["categories"]]
        .drop_duplicates()  # unique categories only
        .to_dict(orient="records")
    )

    # Get list of annotations
    list_annotations = df[required_COCO_keys["annotations"]].to_dict(
        orient="records"
    )

    # Prepare COCO dict
    COCO_dict = {
        "images": list_images,
        "categories": list_categories,
        "annotations": list_annotations,
    }

    # Write to JSON file
    with open(output_filepath, "w") as f:
        json.dump(COCO_dict, f, sort_keys=True, indent=4)

    # Check if valid COCO
    ValidCOCO(output_filepath)

    return output_filepath
