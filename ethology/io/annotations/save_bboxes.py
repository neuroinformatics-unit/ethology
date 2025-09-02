"""Module for exporting manually labelled bounding boxes."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
import pytz
import xarray as xr
from pandera.typing.pandas import DataFrame

from ethology.io.annotations.validate import (
    ValidBBoxesDataFrameCOCO,
    ValidBboxesDataset,
    ValidCOCO,
    _check_input,
    _check_output,
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


@_check_input(validator=ValidBboxesDataset)
@_check_output(validator=ValidCOCO)  # check output is ethology importable
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

    return output_filepath


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
    # Prepare dataframe from xarray dataset
    df_raw = _get_raw_df_from_ds(ds)
    df = _add_COCO_data_to_df(df_raw, ds.attrs)

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
    return df[cols_to_select]


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


def _get_raw_df_from_ds(ds: xr.Dataset) -> pd.DataFrame:
    """Get raw dataframe derived from xarray dataset.

    Parameters
    ----------
    ds : xr.Dataset
        Bounding boxes annotations xarray dataset.

    Returns
    -------
    df : pd.DataFrame
        Wrangled dataframe derived from the xarray dataset.

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

    return df_raw


def _add_COCO_data_to_df(df: pd.DataFrame, ds_attrs: dict) -> pd.DataFrame:
    """Add COCO required data to ds-derived dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset-derived dataframe.
    ds_attrs : dict
        Dataset attributes.

    Returns
    -------
    df : pd.DataFrame
        Dataset-derived dataframe with COCO required data.

    """
    # image
    map_image_id_to_filename = ds_attrs["map_image_id_to_filename"]
    df["image_filename"] = df["image_id"].map(map_image_id_to_filename)

    df["image_width"] = ds_attrs.get("image_width", 0)
    df["image_height"] = ds_attrs.get("image_height", 0)

    # bbox
    df["x_min"] = df["position_x"] - df["shape_x"] / 2
    df["y_min"] = df["position_y"] - df["shape_y"] / 2
    df["width"] = df["shape_x"]
    df["height"] = df["shape_y"]
    df["bbox"] = df[["x_min", "y_min", "width", "height"]].values.tolist()

    df["area"] = df["width"] * df["height"]

    # segmentation as list of lists of coordinates
    # top-left -> top-right -> bottom-right -> bottom-left
    df["segmentation"] = df["bbox"].apply(
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

    # Compute "category" as string from "category_id"
    # rename "category" to "category_id" (in dataset it is an integer)
    map_category_to_str = ds_attrs["map_category_to_str"]
    df.rename(columns={"category": "category_id"}, inplace=True)
    df["category"] = df["category_id"].map(map_category_to_str)
    df["supercategory"] = ""

    # other
    df["iscrowd"] = 0

    # Set index name and add "annotation_id" as column
    df.index.name = "annotation_id"
    df["annotation_id"] = df.index

    # Sort by "image_filename" and remove duplicates
    df = df.sort_values(by=["image_filename"])
    df = df.loc[
        df.astype(str).drop_duplicates(ignore_index=True).index
    ]  # need to serialise lists first before dropping duplicates

    return df
