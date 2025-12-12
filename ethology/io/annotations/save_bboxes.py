"""Save ``ethology`` bounding box annotations datasets to various formats."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pandera.pandas as pa
import pytz
import xarray as xr
from pandera.typing.pandas import DataFrame

from ethology.validators.annotations import (
    ValidBboxAnnotationsCOCO,
    ValidBboxAnnotationsDataset,
    ValidCOCO,
)
from ethology.validators.utils import _check_input, _check_output


@_check_input(validator=ValidBboxAnnotationsDataset)
@_check_output(validator=ValidCOCO)  # check output is ethology-importable
def to_COCO_file(dataset: xr.Dataset, output_filepath: str | Path):
    """Save an ``ethology`` bounding box annotations dataset to a COCO file.

    Parameters
    ----------
    dataset : xarray.Dataset
        Bounding boxes annotations xarray dataset.
    output_filepath : str or pathlib.Path
        Path for the output COCO file.

    Returns
    -------
    str
        Path for the output COCO file.

    Examples
    --------
    Save annotations to a COCO file:

    >>> from ethology.io.annotations import save_bboxes
    >>> save_bboxes.to_COCO_file(ds, "path/to/output_file.json")

    """
    # Compute valid COCO dataframe from xarray dataset
    df = _to_COCO_exportable_df(dataset)

    # Create COCO dictionary from dataframe and export
    COCO_dict = _create_COCO_dict(df)
    with open(output_filepath, "w") as f:
        json.dump(COCO_dict, f, sort_keys=True, indent=2)

    return output_filepath


@_check_input(validator=ValidBboxAnnotationsDataset)
@pa.check_types
def _to_COCO_exportable_df(
    ds: xr.Dataset,
) -> DataFrame[ValidBboxAnnotationsCOCO]:
    """Convert dataset of bounding boxes annotations to a COCO-exportable df.

    The returned dataframe is validated using ValidBBoxesDataFrameCOCO.

    Parameters
    ----------
    ds : xr.Dataset
        A valid dataset of bounding boxes annotations.

    Returns
    -------
    df : pd.DataFrame
        A dataframe of bounding boxes annotations exportable to COCO.

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


@_check_input(validator=ValidBboxAnnotationsDataset)
def _get_raw_df_from_ds(ds: xr.Dataset) -> pd.DataFrame:
    """Get preliminary dataframe from a dataset of bounding boxes annotations.

    If the dataset has an "image_shape" array, the returned dataframe
    will have "image_shape_x" and "image_shape_y" columns.

    The returned dataframe is not COCO-exportable.

    Parameters
    ----------
    ds : xr.Dataset
        A valid dataset of bounding boxes annotations.

    Returns
    -------
    df : pd.DataFrame
        A preliminary dataframe of bounding boxes annotations.

    """
    # Create dataframe from xarray dataset
    df_raw = ds.to_dataframe(dim_order=["image_id", "id", "space"])
    df_raw = df_raw.reset_index()

    # Remove rows where position or shape data is nan
    # (where at least one of the specified columns contains a NaN value.)
    df_raw = df_raw.dropna(subset=["position", "shape"])

    # Pivot the dataframe to get position_x, position_y, shape_x, shape_y, etc.
    # pivot_values: variables with x and y values
    # index_cols: variables **without** x and y values
    pivot_values = [
        c for c in ["position", "shape", "image_shape"] if c in df_raw.columns
    ]
    index_cols = [
        c for c in df_raw.columns if c not in {*pivot_values, "space"}
    ]

    df_raw = df_raw.pivot_table(
        index=index_cols,
        columns="space",
        values=pivot_values,
    ).reset_index()

    # Flatten the columns
    df_raw.columns = [
        "_".join(col).strip() if col[1] != "" else col[0]
        for col in df_raw.columns.values
    ]

    # Reset type for image_shape columns if present
    if all(
        col in df_raw.columns for col in ["image_shape_x", "image_shape_y"]
    ):
        df_raw["image_shape_x"] = df_raw["image_shape_x"].astype(int)
        df_raw["image_shape_y"] = df_raw["image_shape_y"].astype(int)

    return df_raw


@pa.check_types
def _add_COCO_data_to_df(
    df: pd.DataFrame, ds_attrs: dict
) -> DataFrame[ValidBboxAnnotationsCOCO]:
    """Add COCO-required data to preliminary dataframe.

    The input dataframe is obtained from a dataset of bounding boxes
    annotations using ``_get_raw_df_from_ds`` and is not COCO-exportable.

    Parameters
    ----------
    df : pd.DataFrame
        Preliminary dataframe of bounding boxes annotations derived
        from a dataset of bounding boxes annotations.
    ds_attrs : dict
        Attributes of the dataset of bounding boxes annotations.

    Returns
    -------
    df : pd.DataFrame
        COCO-exportable dataframe of bounding boxes annotations.
        The dataframe has the following columns:
        'id', 'annotation_id',
        'image_filename', 'image_id', 'image_width', 'image_height',
        'position_x', 'position_y', 'shape_x', 'shape_y',
        'x_min',  'y_min', 'width', 'height',
        'bbox', 'area', 'segmentation',
        'category', 'supercategory', 'category_id', 'iscrowd'.

    Notes
    -----
    The 'id' column holds the annotation ID per frame, whereas
    the 'annotation_id' column holds the annotation ID across the
    whole dataset.

    """
    # image filename
    map_image_id_to_filename = ds_attrs["map_image_id_to_filename"]
    df["image_filename"] = df["image_id"].map(map_image_id_to_filename)

    # image width and height
    if all(col in df.columns for col in ["image_shape_x", "image_shape_y"]):
        df = df.rename(
            columns={
                "image_shape_x": "image_width",
                "image_shape_y": "image_height",
            },
        )
    else:
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

    # Rename "category" to "category_id"
    # (in input dataset "category" is an integer, but in COCO it is a str)
    df.rename(columns={"category": "category_id"}, inplace=True)
    # and compute "category" as a string from "category_id"
    map_category_to_str = ds_attrs["map_category_to_str"]
    df["category"] = df["category_id"].map(
        lambda x: map_category_to_str.get(x, "")
    )  # set value to "" if category ID is not defined in map_category_to_str

    # Set supercategory to empty string if not defined
    if "supercategory" not in df.columns:
        df["supercategory"] = ""
    else:
        df["supercategory"] = df["supercategory"].astype(str)

    # Set iscrowd always to 0
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


@pa.check_types
def _create_COCO_dict(
    df: DataFrame[ValidBboxAnnotationsCOCO],
) -> dict:
    """Extract COCO dictionary from a COCO-exportable dataframe.

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
    map_columns_to_COCO_fields = (
        ValidBboxAnnotationsCOCO.map_df_columns_to_COCO_fields()
    )
    for sections in ["images", "categories", "annotations"]:
        # Extract and rename required columns for this section
        list_required_columns = map_columns_to_COCO_fields[sections].keys()
        df_section = df[list_required_columns].copy()
        df_section = df_section.rename(
            columns=map_columns_to_COCO_fields[sections]
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
