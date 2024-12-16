"""Module for reading and writing manually labelled annotations."""

import json
from pathlib import Path

import pandas as pd
from movement.validators.files import ValidFile

from ethology.annotations.json_schemas import COCO_SCHEMA, VIA_SCHEMA
from ethology.annotations.validators import (
    ValidCOCOJSON,
    ValidJSON,
    ValidVIAJSON,
)

STANDARD_DF_COLUMNS = [
    "annotation_id",
    "image_filename",
    "image_id",
    "x_min",
    "y_min",
    "width",
    "height",
    "supercategory",
    "category",
]


def df_from_via_json_file(file_path: Path) -> pd.DataFrame:
    """Validate and read untracked VIA JSON file.

    The data is formatted as an untracked annotations DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the untracked VIA JSON file.

    Returns
    -------
    pd.DataFrame
        Untracked annotations DataFrame.

    """
    # Run validators
    file = ValidFile(
        file_path, expected_permission="r", expected_suffix=[".json"]
    )
    json_file = ValidJSON(path=file.path, schema=VIA_SCHEMA)
    via_untracked_file = ValidVIAJSON(json_file.path)

    # Read as standard dataframe
    return _df_from_validated_via_json_file(via_untracked_file.path)


def df_from_coco_json_file(file_path: Path) -> pd.DataFrame:
    """Validate and read untracked COCO JSON file.

    The data is formatted as an untracked annotations DataFrame.

    Parameters
    ----------
    file_path : Path
        Path to the untracked COCO JSON file.

    Returns
    -------
    pd.DataFrame
        Untracked annotations DataFrame.

    """
    # Run validators
    file = ValidFile(
        file_path, expected_permission="r", expected_suffix=[".json"]
    )
    json_file = ValidJSON(path=file.path, schema=COCO_SCHEMA)
    coco_untracked_file = ValidCOCOJSON(json_file.path)

    # Read as standard dataframe
    return _df_from_validated_coco_json_file(coco_untracked_file.path)


def _df_from_validated_via_json_file(file_path):
    """Read VIA JSON file as standard untracked annotations DataFrame."""
    # Read validated json as dict
    with open(file_path) as file:
        data_dict = json.load(file)

    # Prepare data
    image_metadata_dict = data_dict["_via_img_metadata"]
    via_image_id_list = data_dict[
        "_via_image_id_list"
    ]  # ordered list of the keys in image_metadata_dict

    # map filename to keys in image_metadata_dict
    map_filename_to_via_img_id = {
        img_dict["filename"]: ky
        for ky, img_dict in image_metadata_dict.items()
    }

    # Build standard dataframe
    list_rows = []
    # loop thru images
    for _, img_dict in image_metadata_dict.items():
        # loop thru annotations in the image
        for region in img_dict["regions"]:
            region_shape = region["shape_attributes"]
            region_attributes = region["region_attributes"]

            row = {
                "image_filename": img_dict["filename"],
                "x_min": region_shape["x"],
                "y_min": region_shape["y"],
                "width": region_shape["width"],
                "height": region_shape["height"],
                "supercategory": list(region_attributes.keys())[
                    0
                ],  # takes first key as supercategory
                "category": region_attributes[
                    list(region_attributes.keys())[0]
                ],
            }

            # append annotations to df
            list_rows.append(row)

    df = pd.DataFrame(
        list_rows,
        # columns=list(row.keys()),  # do I need this?
    )

    # add image_id column
    df["image_id"] = df["image_filename"].apply(
        lambda x: via_image_id_list.index(map_filename_to_via_img_id[x])
    )

    # add annotation_id column based on index
    df["annotation_id"] = df.index

    # reorder columns to match standard
    df = df.reindex(columns=STANDARD_DF_COLUMNS)

    return df


def _df_from_validated_coco_json_file(file_path: Path) -> pd.DataFrame:
    """Read COCO JSON file as standard untracked annotations DataFrame."""
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

    df = pd.DataFrame(list_rows)
    df.reindex(columns=STANDARD_DF_COLUMNS)

    return df
