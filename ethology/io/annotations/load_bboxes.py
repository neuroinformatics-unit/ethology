"""Module for reading and writing manually labelled annotations."""

import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import xarray as xr

from ethology.io.annotations.validate import ValidCOCO, ValidVIA

# TODO: use pandera
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
    """Read input annotation files as a bboxes xarray dataset.

    Parameters
    ----------
    file_paths : Path | str | list[Path | str]
        Path or list of paths to the input annotation files.
    format : Literal["VIA", "COCO"]
        Format of the input annotation files.
    images_dirs : Path | str | list[Path | str], optional
        Path or list of paths to the directories containing the images the
        annotations refer to. The paths are added to the dataset
        attributes.

    Returns
    -------
    xr.Dataset
        An annotations dataset holding bounding boxes data. The dataset has the
        following dimensions: "image_id", "space", "id".
        - The "image_id" is assigned based on the alphabetically sorted list
        of unique image filenames across all input files.
        -The "space" dimension holds the "x" or "y" coordinates.
        - The "id" dimension corresponds to the annotation ID per image and
        it is assigned from 0 to the max number of annotations per image in
        the full dataset.

        The dataset consists of three arrays:
        - "position", with dimensions (image_id, space, id)
        - "shape", with dimensions (image_id, space, id)
        - "category", with dimensions (image_id, id)
        The "category" array holds category IDs as 1-based integers,
        matching the category IDs in the input file.

        The dataset attributes include:
        - "annotation_files": a list of paths to the input annotation files
        - "annotation_format": the format of the input annotation files
        - "map_category_to_str": a map from category ID to category name
        - "map_image_id_to_filename": a map from image ID to image filename
        - "images_directories": a list of directory paths for the images
        the annotations refer to (optional)


    Notes
    -----
    We use the sorted list of unique image filenames to assign IDs to images,
    so if two images have the same filename but are in different input
    annotation files, they will be assigned the same image ID and their
    annotations will be merged.

    If this behaviour is not desired, and you would like to assign different
    image IDs to images that have the same filename and appear in different
    input annotation files, you can either make the image filenames distinct
    before loading the data, or you can load the data from each file as a
    separate xarray dataset, and then concatenate them as desired.

    Examples
    --------
    Load annotations from two files following VIA format:
    >>> ds = from_files(
    ...     file_paths=[
    ...         "path/to/annotation_file_1.json",
    ...         "path/to/annotation_file_2.json",
    ...     ],
    ...     format="VIA",
    ...     images_dirs=["path/to/images_dir_1", "path/to/images_dir_2"],
    ... )

    """
    # Compute intermediate dataframe df
    if isinstance(file_paths, list):
        df_all = _df_from_multiple_files(file_paths, format=format)
    else:
        df_all = _df_from_single_file(file_paths, format=format)

    # Get map from "image_id" to image_filename
    mapping_df = df_all[["image_filename", "image_id"]].drop_duplicates()
    map_image_id_to_filename = mapping_df.set_index("image_id").to_dict()[
        "image_filename"
    ]

    # Get map from "category_id" to category name
    map_category_to_str = (
        df_all[["category_id", "category"]]
        .drop_duplicates()
        .set_index("category_id")
        .to_dict()["category"]
    )

    # Convert dataframe to xarray dataset
    ds = _df_to_xarray_ds(df_all)

    # Add metadata to the xarraydataset
    ds.attrs = {
        "annotation_files": file_paths,
        "annotation_format": format,
        "images_directories": images_dirs,
        "map_category_to_str": map_category_to_str,
        "map_image_id_to_filename": map_image_id_to_filename,
    }

    return ds


def _df_from_multiple_files(
    list_filepaths: list[Path | str], format: Literal["VIA", "COCO"]
):
    """Read bounding boxes annotations from multiple files as a dataframe.

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
        _df_from_single_file(file_path=file, format=format)
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


def _df_from_single_file(
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

    # Cast category_id for VIA files from str to int if possible,
    # otherwise factorize it
    if format == "VIA" and not df["category_id"].isna().all():
        df = _category_id_as_int(df)

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
    for annot_id, annot_dict in enumerate(data_dict["annotations"]):
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
            "annotation_id": annot_id,
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
            "category_id": category_id,
            # in COCO files, the category_id is always a 1-based integer
        }

        list_rows.append(row)

    return list_rows


def _category_id_as_int(df: pd.DataFrame) -> pd.DataFrame:
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
        # Factorise to 0-based integers
        df["category_id"] = df["category"].factorize(sort=True)[0]
        # Add 1 to the factorised values to make them 1-based
        # (0 is reserved for the background class)
        # (-1 is reserved for missing values in factorisation)
        df.loc[df["category_id"] >= 0, "category_id"] += 1
    return df


def _df_to_xarray_ds(df: pd.DataFrame) -> xr.Dataset:
    """Convert bounding boxes annotations dataframe to an xarray dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Bounding boxes annotations dataframe.

    Returns
    -------
    xr.Dataset
        an xarray dataset with the following dimensions:
        - "image_id": holds the 0-based index of the image in the "images"
        list of the COCO JSON file;
        - "space": "x" or "y";
        - "id": annotation ID per image, assigned from 0 to the max number of
        annotations per image in the full dataset.

        The dataset is made up of the following arrays:
        - position: (image_id, space, id)
        - shape: (image_id, space, id)
        - category: (image_id, id)

    """
    # Compute max number of annotations per image
    max_annotations_per_image = df["image_id"].value_counts().max()

    # Sort the dataframe by image_id
    # Note: the input annotation ID is unique across the dataframe
    df = df.sort_values(by=["image_id"])

    # Compute indices of the rows where the image ID switches
    bool_id_diff_from_prev = df["image_id"].ne(df["image_id"].shift())
    indices_id_switch = np.argwhere(bool_id_diff_from_prev)[1:, 0]

    # Stack position, shape and confidence arrays along ID axis
    map_key_to_columns = {
        "position_array": ["x_min", "y_min"],
        "shape_array": ["width", "height"],
        "category_array": ["category_id"],
    }
    map_key_to_padding = {
        "position_array": (np.float64, np.nan),
        "shape_array": (np.float64, np.nan),
        "category_array": (int, -1),
    }
    array_dict = {}
    for key in map_key_to_columns:
        # extract annotations per image
        list_arrays = np.split(
            df[map_key_to_columns[key]].to_numpy(
                dtype=map_key_to_padding[key][0]  # type: ignore
            ),
            indices_id_switch,  # indices along axis=0
        )  # each array: (n_annotations, N_DIM)

        # pad arrays with NaN values along the annotation ID axis
        list_arrays_padded = [
            np.pad(
                arr,
                ((0, max_annotations_per_image - arr.shape[0]), (0, 0)),
                constant_values=map_key_to_padding[key][1],  # type: ignore
            )
            for arr in list_arrays
        ]  # each array: (n_max_annotations, N_DIM)

        # stack along the first axis (image_id)
        array_dict[key] = np.stack(
            list_arrays_padded, axis=0
        )  # (n_images, n_max_annotations, N_DIM)

        # reorder axes if required
        if "category" not in key:
            array_dict[key] = np.moveaxis(array_dict[key], -1, 1)

    # ----
    # Modify x_min and y_min to represent the bbox centre
    array_dict["position_array"] += array_dict["shape_array"] / 2

    # Create xarray dataset
    return xr.Dataset(
        data_vars=dict(
            position=(
                ["image_id", "space", "id"],
                array_dict["position_array"],
            ),
            shape=(["image_id", "space", "id"], array_dict["shape_array"]),
            category=(
                ["image_id", "id"],
                array_dict["category_array"].squeeze(axis=-1),
            ),
        ),
        coords=dict(
            image_id=df["image_id"].unique(),
            space=["x", "y"],
            id=range(max_annotations_per_image),
            # annotation ID per frame; could be consistent across frames
            # or not
        ),
    )
