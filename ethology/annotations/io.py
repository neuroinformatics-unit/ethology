"""Module for reading and writing manually labelled annotations."""

import json
from pathlib import Path

import pandas as pd
from movement.validators.files import ValidFile

from ethology.annotations.validators import ValidJSON, ValidVIAUntrackedJSON

STANDARD_DF_COLUMNS = [
    "annotation_id",
    "image_filename",
    "image_id",
    "x_min",
    "y_min",
    "width",
    "height",
    "superclass",
    "class",
]


def df_from_via_json_file(file_path: Path):
    """Validate and read untracked VIA JSON file.

    The data is formated as an untracked annotations DataFrame.
    """
    # General file validation
    file = ValidFile(
        file_path, expected_permission="r", expected_suffix=[".json"]
    )

    # JSON file validation
    json_file = ValidJSON(file.path)

    # VIA Untracked JSON schema validation
    via_untracked_file = ValidVIAUntrackedJSON(json_file.path)

    # Read as standard dataframe
    return _df_from_validated_via_json_file(via_untracked_file.path)


def _df_from_validated_via_json_file(file_path):
    """Read VIA JSON file as standard untracked annotations DataFrame."""
    # Read validated json as dict
    with open(file_path) as file:
        data_dict = json.load(file)

    # Get relevant fields
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

            # append annotations to df
            list_rows.append(
                {
                    "image_filename": img_dict["filename"],
                    "x_min": region_shape["x"],
                    "y_min": region_shape["y"],
                    "width": region_shape["width"],
                    "height": region_shape["height"],
                    "superclass": list(region_attributes.keys())[
                        0
                    ],  # takes first key as superclass
                    "class": region_attributes[
                        list(region_attributes.keys())[0]
                    ],
                },
            )

    df = pd.DataFrame(
        list_rows,
        columns=[
            col for col in STANDARD_DF_COLUMNS if not col.endswith("_id")
        ],
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
