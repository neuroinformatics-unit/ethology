"""Module for exporting manually labelled bounding boxes."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandera.pandas as pa
import pytz
import xarray as xr
from pandera.typing.pandas import DataFrame

from ethology.io.annotations.validate import (
    ValidBBoxesDataFrameCOCO,
    ValidCOCO,
)

# Mapping of dataframe columns to COCO fields
MAP_COLUMNS_TO_COCO_FIELDS = {
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
        # "confidence": "score",  # only for predictions
    },
}


@pa.check_types
def _to_COCO_exportable_df(
    ds: xr.Dataset,
) -> DataFrame[ValidBBoxesDataFrameCOCO]:
    """Convert annotations xarray dataset to a COCO exportable dataframe.

    The returned dataframe is validated using ValidBBoxesDataFrameCOCO.

    Parameters
    ----------
    ds : xr.Dataset
        Bounding boxes annotations xarray dataset.

    Returns
    -------
    df : pd.DataFrame
        A valid dataframe of bounding boxes annotations exportable to COCO.

    """
    # Create dataframe from xarray dataset
    df_raw = ds.to_dataframe(dim_order=["image_id", "id", "space"])
    df_raw = df_raw.reset_index()

    # Remove rows where position or shape data is nan
    # (where at least one of the specified columns contains a NaN value.)
    df_raw = df_raw.dropna(subset=["position", "shape"])

    # Add "category" column if not present
    if "category" not in df_raw.columns:
        df_raw["category"] = -1

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
    # df_raw.rename(columns={"id": "id_per_frame"}, inplace=True)
    df_raw.rename(columns={"category": "category_id"}, inplace=True)

    # Compute bbox coordinates for each annotation
    df_raw["x_min"] = df_raw["position_x"] - df_raw["shape_x"] / 2
    df_raw["y_min"] = df_raw["position_y"] - df_raw["shape_y"] / 2
    df_raw["width"] = df_raw["shape_x"]
    df_raw["height"] = df_raw["shape_y"]
    df_raw["bbox"] = df_raw[
        ["x_min", "y_min", "width", "height"]
    ].values.tolist()

    # Compute "image_filename" from "image_id"
    map_image_id_to_filename = ds.attrs["map_image_id_to_filename"]
    df_raw["image_filename"] = df_raw["image_id"].map(map_image_id_to_filename)

    # Compute "category" as string from "category_id"
    map_category_to_str = ds.attrs["map_category_to_str"]
    df_raw["category"] = df_raw["category_id"].map(map_category_to_str)

    # Compute "area" for each annotation
    df_raw["area"] = df_raw["width"] * df_raw["height"]

    # Compute "segmentation" for each annotation
    # top-left -> top-right -> bottom-right -> bottom-left
    df_raw["segmentation"] = df_raw["bbox"].apply(
        lambda bbox: [
            [
                bbox[0],  # top-left x
                bbox[1],  # top-left y
                bbox[0] + bbox[2],  # top-right x
                bbox[1],  # top-right y
                bbox[0] + bbox[2],  # bottom-right x
                bbox[1] + bbox[3],  # bottom-right y
                bbox[0],  # bottom-left x
                bbox[1] + bbox[3],  # bottom-left y
            ]
        ]
    )

    # Fill in default values
    df_raw["iscrowd"] = 0
    df_raw["supercategory"] = ""
    df_raw["image_width"] = ds.attrs.get("image_width", 0)
    df_raw["image_height"] = ds.attrs.get("image_height", 0)

    # Set index name
    df_raw.index.name = "annotation_id"
    df_raw["annotation_id"] = df_raw.index

    # Select columns to keep
    cols_to_select = [
        "annotation_id",
        "image_id",
        "image_filename",
        "image_width",
        "image_height",
        "bbox",
        "area",
        "segmentation",
        "category",  # str
        "category_id",  # int
        "supercategory",
        "iscrowd",
    ]
    df_output = df_raw[cols_to_select]

    # Sort by "image_filename" and remove duplicates
    df_output = df_output.sort_values(by=["image_filename"])
    df_output = df_output.loc[
        df_output.astype(str).drop_duplicates(ignore_index=True).index
    ]

    return df_output


@pa.check_types
def _create_COCO_dict(df: DataFrame[ValidBBoxesDataFrameCOCO]) -> dict:
    """Extract COCO dictionary from a COCO exportable dataframe.

    Parameters
    ----------
    df : DataFrame[ValidBBoxesDataFrameCOCO]
        COCO exportable dataframe.

    Returns
    -------
    COCO_dict : dict
        COCO dictionary.

    """
    COCO_dict: dict[str, Any] = {}
    for sections in ["images", "categories", "annotations"]:
        # Extract and rename required columns for this section
        list_required_columns = MAP_COLUMNS_TO_COCO_FIELDS[sections].keys()
        df_section = df[list_required_columns].copy()
        df_section = df_section.rename(
            columns=MAP_COLUMNS_TO_COCO_FIELDS[sections]
        )

        # Extract rows as lists of dictionaries
        if sections == "annotations":
            row_dicts = df_section.to_dict(orient="records")
        else:
            row_dicts = df_section.drop_duplicates().to_dict(orient="records")

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
    # Compute valid COCO dataframe from xarray dataset
    df = _to_COCO_exportable_df(ds)

    # Create COCO dictionary from dataframe and export
    COCO_dict = _create_COCO_dict(df)
    with open(output_filepath, "w") as f:
        json.dump(COCO_dict, f, sort_keys=True, indent=2)

    # Check output JSON file is importable by ethology
    ValidCOCO(output_filepath)

    return output_filepath
