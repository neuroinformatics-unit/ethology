"""Module for reading and writing manually labelled annotations."""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import pandas as pd

from ethology.annotations.validators import ValidCOCO, ValidVIA

# definition of standard bboxes dataframe
STANDARD_BBOXES_DF_INDEX = "annotation_id"
STANDARD_BBOXES_DF_COLUMNS = [
    "image_filename",
    "image_id",
    "x_min",
    "y_min",
    "width",
    "height",
    "supercategory",
    "category",
    "image_width",
    "image_height",
]  # if a column is not defined, it is filled with nan


def df_bboxes_from_files(
    file_paths: Path | list[Path],
    format: Literal["VIA", "COCO"],
    images_dirs: Path | list[Path] | None = None,
    **kwargs,
) -> pd.DataFrame:
    """Read bounding boxes annotations as a dataframe.

    Parameters
    ----------
    file_paths : Path | list[Path]
        Path or list of paths to the input annotation files.
    format : Literal["VIA", "COCO"]
        Format of the input annotation files.
    images_dirs : Path | list[Path], optional
        Path or list of paths to the directories containing the images the
        annotations refer to.
    **kwargs
        Additional keyword arguments to pass to the
        ``pandas.DataFrame.drop_duplicates`` method. The ``ignore_index=True``
        argument is always applied to force an index reset, and the ``inplace``
        argument is set to ``False`` and cannot be overridden. The settings
        apply if one or multiple files are read.

    Returns
    -------
    pd.DataFrame
        Bounding boxes annotations dataframe. The dataframe is indexed
        by "annotation_id" and has the following columns: "image_filename",
        "image_id", "image_width", "image_height", "x_min", "y_min",
        "width", "height", "supercategory", "category".

    See Also
    --------
    pandas.concat : Concatenate pandas objects along a particular axis.

    pandas.DataFrame.drop_duplicates : Return DataFrame with duplicate rows
    removed.

    """
    # Check kwargs that are forwarded to drop_duplicates
    for fixed_kwargs in ["ignore_index", "inplace"]:
        if fixed_kwargs in kwargs:
            raise ValueError(
                f"The '{fixed_kwargs}' argument for "
                "`pandas.DataFrame.drop_duplicates` may not be overridden."
            )

    if isinstance(file_paths, list):
        # Read multiple files
        df_all = _df_bboxes_from_multiple_files(
            file_paths, format=format, **kwargs
        )

    else:
        # Read single file
        df_all = _df_bboxes_from_single_file(
            file_paths, format=format, **kwargs
        )

    # Add metadata
    df_all.metadata = {
        "input_files": file_paths,
        "format": format,
        "images_dirs": images_dirs,
    }

    return df_all


def _df_bboxes_from_multiple_files(
    list_filepaths: list[Path], format: Literal["VIA", "COCO"], **kwargs
):
    """Read bounding boxes annotations from multiple files.

    Parameters
    ----------
    list_filepaths : list[Path]
        List of paths to the input annotation files
    format : Literal["VIA", "COCO"]
        Format of the input annotation files.
        Currently supported formats are "VIA" and "COCO".
    **kwargs
        Additional keyword arguments to pass to the
        ``pandas.DataFrame.drop_duplicates`` method. The ``ignore_index=True``
        argument is always applied to force an index reset, and the ``inplace``
        argument is set to ``False`` and cannot be overridden. The settings
        apply if one or multiple files are read.

    Returns
    -------
    pd.DataFrame
        Bounding boxes annotations dataframe. The dataframe is indexed
        by "annotation_id" and has the following columns: "image_filename",
        "image_id", "image_width", "image_height", "x_min", "y_min",
        "width", "height", "supercategory", "category".

    """
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
    df_all = df_all.drop_duplicates(ignore_index=True, inplace=False, **kwargs)

    # Set the index name to "annotation_id"
    df_all.index.name = STANDARD_BBOXES_DF_INDEX

    return df_all


def _df_bboxes_from_single_file(
    file_path: Path, format: Literal["VIA", "COCO"], **kwargs
) -> pd.DataFrame:
    """Read bounding boxes annotations from a single file.

    Parameters
    ----------
    file_path : Path
        Path to the input annotation file.
    format : Literal["VIA", "COCO"]
        Format of the input annotation file.
        Currently supported formats are "VIA" and "COCO".
    **kwargs
        Additional keyword arguments to pass to the
        ``pandas.DataFrame.drop_duplicates`` method. The ``ignore_index=True``
        argument is always applied to force an index reset, and the ``inplace``
        argument is set to ``False`` and cannot be overridden. The settings
        apply if one or multiple files are read.

    Returns
    -------
    pd.DataFrame
        Bounding boxes annotations dataframe. The dataframe is indexed
        by "annotation_id" and has the following columns: "image_filename",
        "image_id", "image_width", "image_height", "x_min", "y_min",
        "width", "height", "supercategory", "category".

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
    """Read bounding boxes annotations from a single specific file.

    Parameters
    ----------
    file_path : Path
        Path to the input annotation file.
    validator : type[ValidVIA] | type[ValidCOCO]
        Validator class for the input annotation file.
    get_rows_from_file : Callable
        Function to extract rows from the validated input annotation file.
    **kwargs
        Additional keyword arguments to pass to the
        ``pandas.DataFrame.drop_duplicates`` method. The ``ignore_index=True``
        argument is always applied to force an index reset, and the ``inplace``
        argument is set to ``False`` and cannot be overridden. The settings
        apply if one or multiple files are read.

    Returns
    -------
    pd.DataFrame
        Bounding boxes annotations dataframe. The dataframe is indexed
        by "annotation_id" and has the following columns: "image_filename",
        "image_id", "image_width", "image_height", "x_min", "y_min",
        "width", "height", "supercategory", "category".

    """
    # Validate file
    valid_file = validator(file_path)

    # Build dataframe from extracted rows
    list_rows = get_rows_from_file(valid_file.path)
    df = pd.DataFrame(list_rows)

    # Set "annotation_id" as index
    # (otherwise duplicate annotations are not identified as such)
    df = df.set_index(STANDARD_BBOXES_DF_INDEX)

    # Drop duplicates and reset indices.
    # We use ignore_index=True so that the resulting axis is labeled 0,1,…,n-1.
    # NOTE: after this the index name is no longer "annotation_id"
    df = df.drop_duplicates(ignore_index=True, inplace=False, **kwargs)

    # Reorder columns to match standard columns
    df = df.reindex(columns=STANDARD_BBOXES_DF_COLUMNS)

    # Set the index name to "annotation_id"
    df.index.name = STANDARD_BBOXES_DF_INDEX

    # Read as standard dataframe
    return df


def _df_rows_from_valid_VIA_file(file_path: Path) -> list[dict]:
    """Extract list of rows from a validated VIA JSON file.

    Parameters
    ----------
    file_path : Path
        Path to the validated VIA JSON file.

    Returns
    -------
    list[dict]
        List of rows extracted from the validated VIA JSON file.

    """
    # Read validated json as dict
    with open(file_path) as file:
        data_dict = json.load(file)

    # Prepare data
    image_metadata_dict = data_dict["_via_img_metadata"]
    via_image_id_list = data_dict["_via_image_id_list"]
    via_attributes = data_dict["_via_attributes"]
    supercategories_props = {}
    if "region" in via_attributes:
        supercategories_props = via_attributes["region"]

    # Map image filenames to the image keys used by VIA
    # the VIA keys are <filename><filesize> strings
    map_filename_to_via_img_id = {
        img_dict["filename"]: ky
        for ky, img_dict in image_metadata_dict.items()
    }

    # Get list of rows in dataframe
    list_rows = []
    annotation_id = 0
    # loop through images
    for _, img_dict in image_metadata_dict.items():
        # loop thru annotations in the image
        for region in img_dict["regions"]:
            # Extract region data
            region_shape = region["shape_attributes"]
            region_attributes = region["region_attributes"]

            # Define supercategory and category.
            # We take first key in "region_attributes" as the supercategory,
            # and its value as category_id_str
            if region_attributes and supercategories_props:
                supercategory = sorted(list(region_attributes.keys()))[0]
                category_id_str = region_attributes[supercategory]
                category = supercategories_props[supercategory]["options"][
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

            list_rows.append(row)

            # update "annotation_id"
            annotation_id += 1

    return list_rows


def _df_rows_from_valid_COCO_file(file_path: Path) -> list[dict]:
    """Extract list of rows from a validated COCO JSON file.

    Parameters
    ----------
    file_path : Path
        Path to the validated COCO JSON file.

    Returns
    -------
    list[dict]
        List of rows extracted from the validated COCO JSON file.

    """
    # Read validated json as dict
    with open(file_path) as file:
        data_dict = json.load(file)

    # Prepare data
    map_image_id_to_filename = {
        img_dict["id"]: img_dict["file_name"]
        for img_dict in data_dict["images"]
    }
    map_image_id_to_width_height = {
        img_dict["id"]: (img_dict["width"], img_dict["height"])
        for img_dict in data_dict["images"]
    }

    map_category_id_to_category_data = {
        cat_dict["id"]: (cat_dict["name"], cat_dict["supercategory"])
        for cat_dict in data_dict["categories"]
    }  # category data: category name, supercategor name

    # Build standard dataframe
    list_rows = []
    for annot_dict in data_dict["annotations"]:
        annotation_id = annot_dict["id"]

        # image data
        image_id = annot_dict["image_id"]
        image_filename = map_image_id_to_filename[image_id]
        image_width = map_image_id_to_width_height[image_id][0]
        image_height = map_image_id_to_width_height[image_id][1]

        # bbox data
        x_min, y_min, width, height = annot_dict["bbox"]

        # category data
        category_id = annot_dict["category_id"]
        category, supercategory = map_category_id_to_category_data[category_id]

        row = {
            "annotation_id": annotation_id,
            "image_filename": image_filename,
            "image_id": image_id,
            "image_width": image_width,
            "image_height": image_height,
            "x_min": x_min,
            "y_min": y_min,
            "width": width,
            "height": height,
            "supercategory": supercategory,
            "category": category,
        }

        list_rows.append(row)

    return list_rows
