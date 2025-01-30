"""Module for reading and writing manually labelled annotations."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd

from ethology.annotations.validators import ValidCOCO, ValidVIA

STANDARD_BBOXES_INDEX = "annotation_id"
STANDARD_BBOXES_COLUMNS = [
    "image_filename",
    "image_id",
    "x_min",
    "y_min",
    "width",
    "height",
    "supercategory",
    "category",
]  # defines minimum columns?


def df_bboxes_from_file(
    file_path: Path | list[Path],
    format: Literal["VIA", "COCO"],
    **kwargs,
) -> pd.DataFrame:
    """Read bounding boxes annotations as a dataframe.

    Parameters
    ----------
    file_path : Path | list[Path]
        Path to the input annotations file or a list of paths.
    format : Literal["VIA", "COCO"]
        Format of the input annotations file.
    **kwargs
        Additional keyword arguments to pass to the
        pandas.DataFrame.drop_duplicates method. The ignore_index=True
        argument is always applied to force an index reset. The settings
        apply if one or multiple files are read.

    Returns
    -------
    pd.DataFrame
        Bounding boxes annotations dataframe. The dataframe has the
        following columns: "annotation_id", "image_filename", "image_id",
        "x_min", "y_min", "width", "height", "supercategory", "category".

    See Also
    --------
    pandas.concat : Concatenate pandas objects along a particular axis.
    pandas.DataFrame.drop_duplicates : Return DataFrame with duplicate rows
    removed.

    """
    if isinstance(file_path, list):
        # Read multiple files
        df_all = _df_bboxes_from_multiple_files(
            file_path, format=format, **kwargs
        )

    else:
        # Read single VIA file
        df_all = _df_bboxes_from_single_file(
            file_path, format=format, **kwargs
        )

    # Add list of input files as metadata
    df_all.metadata = {"input_files": file_path}

    return df_all


def _df_bboxes_from_multiple_files(list_filepaths, format, **kwargs):
    """Read bounding boxes annotations from multiple files."""
    # Get list of dataframes
    df_list = [
        _df_bboxes_from_single_file(file, format=format)
        for file in list_filepaths
    ]

    # Concatenate with ignore_index=True,
    # so that the resulting axis is labeled 0,1,…,n - 1.
    # NOTE: after ignore_index=True the index name is no longer "annotation_id"
    df_all = pd.concat(df_list, ignore_index=True)

    # Update image_id based on the full sorted list of image filenames
    list_image_filenames = sorted(list(df_all["image_filename"].unique()))
    df_all["image_id"] = df_all["image_filename"].apply(
        lambda x: list_image_filenames.index(x)
    )

    # Remove duplicates
    df_all = df_all.drop_duplicates(ignore_index=True, **kwargs)

    # Set the index name to "annotation_id"
    df_all.index.name = STANDARD_BBOXES_INDEX

    return df_all


def _df_bboxes_from_single_file(
    file_path: Path, format: str, **kwargs
) -> pd.DataFrame:
    """Read bounding boxes annotations from a single file.

    Parameters
    ----------
    file_path : Path
        Path to the input annotations file.
    format : str
        Format of the input annotations file.
        Currently supported formats are "VIA" and "COCO".
    **kwargs
        Additional keyword arguments to pass to the
        pandas.DataFrame.drop_duplicates method. The ignore_index=True
        argument is always applied to force an index reset.

    Returns
    -------
    pd.DataFrame
        Bounding boxes annotations dataframe. The dataframe has the
        following columns: "annotation_id", "image_filename", "image_id",
        "x_min", "y_min", "width", "height", "supercategory", "category".

    """
    if format == "VIA":
        return _df_bboxes_from_single_specific_file(
            file_path,
            validator=ValidVIA,
            get_rows_from_file=_df_rows_from_valid_VIA_file,
            **kwargs,
        )
    elif format == "COCO":
        return _df_bboxes_from_single_specific_file(
            file_path,
            validator=ValidCOCO,
            get_rows_from_file=_df_rows_from_valid_COCO_file,
            **kwargs,
        )
    else:
        raise ValueError(f"Unsupported format: {format}")


def _df_bboxes_from_single_specific_file(
    file_path: Path,
    validator: type[ValidVIA] | type[ValidCOCO],
    get_rows_from_file: Callable,
    **kwargs,
) -> pd.DataFrame:
    # Validate file
    valid_file = validator(file_path)

    # Build basic dataframe
    list_rows = get_rows_from_file(valid_file.path)
    df = pd.DataFrame(list_rows)  # , index=STANDARD_BBOXES_INDEX)

    # set annotation_id as index
    # (otherwise duplicate annotations are not detected)
    df = df.set_index(STANDARD_BBOXES_INDEX)

    # drop duplicates and reset indices
    # ignore_index=True so that the resulting axis is labeled 0,1,…,n - 1.
    # NOTE: the index name is no longer "annotation_id"
    df = df.drop_duplicates(ignore_index=True, **kwargs)

    # reorder columns to match standard columns
    df = df.reindex(columns=STANDARD_BBOXES_COLUMNS)

    # Set the index name to "annotation_id"
    df.index.name = STANDARD_BBOXES_INDEX

    # Read as standard dataframe
    return df


def _df_rows_from_valid_VIA_file(file_path: Path) -> list[dict]:
    """Read untracked VIA JSON file as a list of rows."""
    # Read validated json as dict
    with open(file_path) as file:
        data_dict = json.load(file)

    # Prepare data
    image_metadata_dict = data_dict["_via_img_metadata"]
    via_image_id_list = data_dict["_via_image_id_list"]
    supercategories_dict = {}
    if "region" in data_dict["_via_attributes"]:
        supercategories_dict = data_dict["_via_attributes"]["region"]

    # Map image filenames to image_metadata_dict keys
    # the image_metadata_dict keys are <filename><filesize> strings
    map_filename_to_via_img_id = {
        img_dict["filename"]: ky
        for ky, img_dict in image_metadata_dict.items()
    }

    # Get list of rows in dataframe
    list_rows = []
    annotation_id = 0
    # loop thru images
    for _, img_dict in image_metadata_dict.items():
        # loop thru annotations in the image
        for region in img_dict["regions"]:
            region_shape = region["shape_attributes"]
            region_attributes = region["region_attributes"]

            # Define supercategory and category.
            # We take first key of the region_attributes as supercategory,
            # and its value as category_id_str
            if list(region_attributes.keys()) and supercategories_dict:
                supercategory = list(region_attributes.keys())[0]
                category_id_str = region_attributes[supercategory]
                category = supercategories_dict[supercategory]["options"][
                    category_id_str
                ]

            else:
                supercategory = ""
                category = ""

            row = {
                "annotation_id": annotation_id,
                "image_filename": img_dict["filename"],
                "image_id": via_image_id_list.index(
                    map_filename_to_via_img_id[img_dict["filename"]]
                ),  # integer based on the VIA image ID
                "x_min": region_shape["x"],
                "y_min": region_shape["y"],
                "width": region_shape["width"],
                "height": region_shape["height"],
                "supercategory": supercategory,
                "category": category,
            }

            # append annotations to df
            list_rows.append(row)

            # increment annotation_id
            annotation_id += 1

    return list_rows


def _df_rows_from_valid_COCO_file(file_path: Path) -> list[dict]:
    """Read untracked COCO JSON file as a list of rows."""
    # Read validated json as dict
    with open(file_path) as file:
        data_dict = json.load(file)

    # Prepare data
    map_image_id_to_filename = {
        img_dict["id"]: img_dict["file_name"]
        for img_dict in data_dict["images"]
    }

    map_category_id_to_category_data = {
        cat_dict["id"]: (cat_dict["name"], cat_dict["supercategory"])
        for cat_dict in data_dict["categories"]
    }

    # Build standard dataframe
    list_rows = []
    for annot_dict in data_dict["annotations"]:
        annotation_id = annot_dict["id"]
        # image data
        image_id = annot_dict["image_id"]
        image_filename = map_image_id_to_filename[image_id]

        # bbox data
        x_min, y_min, width, height = annot_dict["bbox"]

        # class data
        category_id = annot_dict["category_id"]
        category, supercategory = map_category_id_to_category_data[category_id]

        row = {
            "annotation_id": annotation_id,
            "image_filename": image_filename,
            "image_id": image_id,
            "x_min": x_min,
            "y_min": y_min,
            "width": width,
            "height": height,
            "supercategory": supercategory,
            "category": category,
        }

        list_rows.append(row)

    return list_rows
