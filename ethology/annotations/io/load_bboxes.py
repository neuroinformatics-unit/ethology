"""Module for reading and writing manually labelled annotations."""

import json
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


def from_files(
    file_paths: Path | str | list[Path | str],
    format: Literal["VIA", "COCO"],
    images_dirs: Path | str | list[Path | str] | None = None,
) -> pd.DataFrame:
    """Read input annotation files as a bboxes dataframe.

    Parameters
    ----------
    file_paths : Path | str | list[Path | str]
        Path or list of paths to the input annotation files.
    format : Literal["VIA", "COCO"]
        Format of the input annotation files.
    images_dirs : Path | str | list[Path | str], optional
        Path or list of paths to the directories containing the images the
        annotations refer to.

    Returns
    -------
    pd.DataFrame
        Bounding boxes annotations dataframe. The dataframe is indexed
        by "annotation_id" and has the following columns: "image_filename",
        "image_id", "image_width", "image_height", "x_min", "y_min",
        "width", "height", "supercategory", "category". It also has the
        following attributes: "annotation_files", "annotation_format",
        "images_directories". The "image_id" is assigned based
        on the alphabetically sorted list of unique image filenames across all
        input files.

    Notes
    -----
    We use image filenames' to assign IDs to images, so if two images have the
    same name but are in different input annotation files, they will be
    assigned the same image ID and their annotations will be merged.

    If this behaviour is not desired, and you would like to assign different
    image IDs to images that have the same name but appear in different input
    annotation files, you can either make the image filenames distinct before
    loading the data, or you can load the data from each file
    as a separate dataframe, and then concatenate them as desired.

    See Also
    --------
    pandas.concat : Concatenate pandas objects along a particular axis.

    pandas.DataFrame.drop_duplicates : Return DataFrame with duplicate rows
    removed.

    """
    # Delegate to reader of either a single file or multiple files
    if isinstance(file_paths, list):
        df_all = _from_multiple_files(file_paths, format=format)
    else:
        df_all = _from_single_file(file_paths, format=format)

    # Add metadata
    df_all.attrs = {
        "annotation_files": file_paths,
        "annotation_format": format,
        "images_directories": images_dirs,
    }

    return df_all


def _from_multiple_files(
    list_filepaths: list[Path | str], format: Literal["VIA", "COCO"]
):
    """Read bounding boxes annotations from multiple files.

    Parameters
    ----------
    list_filepaths : list[Path | str]
        List of paths to the input annotation files
    format : Literal["VIA", "COCO"]
        Format of the input annotation files.
        Currently supported formats are "VIA" and "COCO".

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
        _from_single_file(file_path=file, format=format)
        for file in list_filepaths
    ]

    # Concatenate with ignore_index=True,
    # so that the resulting axis is labeled 0,1,…,n - 1.
    # NOTE: after ignore_index=True the index name is no longer "annotation_id"
    df_all = pd.concat(df_list, ignore_index=True)

    # Update "image_id" based on the alphabetically sorted list of unique image
    # filenames across all input files
    list_image_filenames = sorted(list(df_all["image_filename"].unique()))
    df_all["image_id"] = df_all["image_filename"].apply(
        lambda x: list_image_filenames.index(x)
    )

    # Remove duplicates that may exist across files
    df_all = df_all.drop_duplicates(ignore_index=True, inplace=False)

    # Set the index name back to "annotation_id"
    df_all.index.name = STANDARD_BBOXES_DF_INDEX

    return df_all


def _from_single_file(
    file_path: Path | str, format: Literal["VIA", "COCO"]
) -> pd.DataFrame:
    """Read bounding boxes annotations from a single file.

    Parameters
    ----------
    file_path : Path | str
        Path to the input annotation file.
    format : Literal["VIA", "COCO"]
        Format of the input annotation file.
        Currently supported formats are "VIA" and "COCO".

    Returns
    -------
    pd.DataFrame
        Bounding boxes annotations dataframe. The dataframe is indexed
        by "annotation_id" and has the following columns: "image_filename",
        "image_id", "image_width", "image_height", "x_min", "y_min",
        "width", "height", "supercategory", "category".

    """
    # Choose the appropriate validator and row-extraction function
    validator: type[ValidVIA | ValidCOCO]
    if format == "VIA":
        validator = ValidVIA
        get_rows_from_file = _df_rows_from_valid_VIA_file
    elif format == "COCO":
        validator = ValidCOCO
        get_rows_from_file = _df_rows_from_valid_COCO_file
    else:
        raise ValueError(f"Unsupported format: {format}")

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
    df = df.drop_duplicates(ignore_index=True, inplace=False)

    # Reorder columns to match standard columns
    df = df.reindex(columns=STANDARD_BBOXES_DF_COLUMNS)

    # Set the index name to "annotation_id"
    df.index.name = STANDARD_BBOXES_DF_INDEX

    # Read as standard dataframe
    return df


def _df_rows_from_valid_VIA_file(file_path: Path) -> list[dict]:
    """Extract list of dataframe rows from a validated VIA JSON file.

    Parameters
    ----------
    file_path : Path
        Path to the validated VIA JSON file.

    Returns
    -------
    list[dict]
        List of dataframe rows extracted from the validated VIA JSON file.

    """
    # Read validated json as dict
    with open(file_path) as file:
        data_dict = json.load(file)

    # Prepare data
    image_metadata_dict = data_dict["_via_img_metadata"]
    list_sorted_filenames = sorted(
        [img_dict["filename"] for img_dict in image_metadata_dict.values()]
    )

    via_attributes = data_dict["_via_attributes"]
    supercategories_props = {}
    if "region" in via_attributes:
        supercategories_props = via_attributes["region"]

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
                "image_id": list_sorted_filenames.index(img_dict["filename"]),
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
    """Extract list of dataframe rows from a validated COCO JSON file.

    Parameters
    ----------
    file_path : Path
        Path to the validated COCO JSON file.

    Returns
    -------
    list[dict]
        List of dataframe rows extracted from the validated COCO JSON file.

    """
    # Read validated json as dict
    with open(file_path) as file:
        data_dict = json.load(file)

    # Prepare data
    # We define image_id_ethology as the 0-based index of the image in the
    # "images" list of the COCO JSON file. The following assumes the number of
    # unique image_ids in the input COCO file matches the number of elements
    # in the "images" list.
    map_img_id_coco_to_ethology = {
        img_dict["id"]: idx
        for idx, img_dict in enumerate(
            sorted(data_dict["images"], key=lambda x: x["file_name"])
        )
    }
    map_img_id_coco_to_filename = {
        img_dict["id"]: img_dict["file_name"]
        for img_dict in data_dict["images"]
    }
    map_img_id_coco_to_width_height = {
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
        img_id_coco = annot_dict["image_id"]
        image_filename = map_img_id_coco_to_filename[img_id_coco]
        image_width, image_height = map_img_id_coco_to_width_height[
            img_id_coco
        ]

        # compute image ID following ethology convention
        img_id_ethology = map_img_id_coco_to_ethology[img_id_coco]

        # bbox data
        x_min, y_min, width, height = annot_dict["bbox"]

        # category data
        category_id = annot_dict["category_id"]
        category, supercategory = map_category_id_to_category_data[category_id]

        row = {
            "annotation_id": annotation_id,
            "image_filename": image_filename,
            "image_id": img_id_ethology,
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
