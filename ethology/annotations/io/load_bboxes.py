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
    "category_id",
    "image_width",
    "image_height",
]  # superset of columns in the standard dataframe


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
        input files. The "category_id" column is always a 0-based integer, 
        except for VIA files where the values specified in the input file are retained.

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
        "width", "height", "supercategory", "category", "category_id".

    """
    # Get list of dataframes
    df_list = [
        _from_single_file(file_path=file, format=format)
        for file in list_filepaths
    ]

    # Concatenate and reindex
    # the resulting axis is labeled 0,1,…,n - 1.
    # NOTE: after ignore_index=True the index name is no longer "annotation_id"
    df_all = pd.concat(df_list, ignore_index=True)

    # Update "image_id" based on the alphabetically sorted list of unique image
    # filenames across all input files
    list_image_filenames = sorted(list(df_all["image_filename"].unique()))
    df_all["image_id"] = df_all["image_filename"].apply(
        lambda x: list_image_filenames.index(x)
    )

    # Sort by image_filename
    df_all = df_all.sort_values(by=["image_filename"])

    # Remove duplicates that may exist across files and reindex
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
        "width", "height", "supercategory", "category", "category_id".

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

    # Build dataframe from extracted rows
    valid_file = validator(file_path)
    list_rows = get_rows_from_file(valid_file.path)
    df = pd.DataFrame(list_rows)

    # Sort annotations by image_filename
    df = df.sort_values(by=["image_filename"])

    # Drop duplicates and reindex
    # The resulting axis is labeled 0,1,…,n-1.
    df = df.drop_duplicates(
        subset=[col for col in df.columns if col != "annotation_id"],
        ignore_index=True,
        inplace=False,
    )

    # Fix category_id for VIA files if required
    # Cast as an int if possible, otherwise factorize it
    if format == "VIA" and not df["category_id"].isna().all():
        df = _VIA_category_id_as_int(df)

    # Reorder columns to match standard columns
    # If columns dont exist they are filled with nan / na values
    df = df.reindex(columns=STANDARD_BBOXES_DF_COLUMNS + ["annotation_id"])

    # Set the index name to "annotation_id"
    df = df.set_index(STANDARD_BBOXES_DF_INDEX)

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

    # Get supercategories and categories
    supercategories_dict = {}
    if "region" in via_attributes:
        supercategories_dict = via_attributes["region"]

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
            # A region (bbox) can have multiple supercategories.
            # We only consider the first supercategory in alphabetical order.
            if region_attributes and supercategories_dict:
                # bbox data
                supercategory = sorted(list(region_attributes.keys()))[0]
                category_id_str = region_attributes[supercategory]

                # map to category name
                category = supercategories_dict[supercategory]["options"][
                    category_id_str
                ]
            # If not defined, set to None
            else:
                supercategory = None
                category = None
                category_id_str = None

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
                "category_id": category_id_str,
                # in VIA files, the category_id is a string
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
            "category_id": category_id - 1,
            # in COCO files, the category_id is always a 1-based integer
        }

        list_rows.append(row)

    return list_rows


def _VIA_category_id_as_int(df: pd.DataFrame) -> pd.DataFrame:
    """Convert category_id to int if possible, otherwise factorize it.

    Parameters
    ----------
    df : pd.DataFrame
        Bounding boxes annotations dataframe.

    Returns
    -------
    pd.DataFrame
        Bounding boxes annotations dataframe with "category_id" as int.

    """
    try:
        df["category_id"] = df["category_id"].astype(int)
    except ValueError:
        df["category_id"] = df["category"].factorize(sort=True)[0]
    return df
