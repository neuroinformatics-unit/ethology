"""Load bounding boxes annotations into ``ethology``."""

import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import pandera.pandas as pa
import xarray as xr
from pandera.typing.pandas import DataFrame

from ethology.validators.annotations import (
    ValidBboxAnnotationsDataFrame,
    ValidBboxAnnotationsDataset,
    ValidCOCO,
    ValidVIA,
)
from ethology.validators.utils import _check_output


@_check_output(ValidBboxAnnotationsDataset)
def from_files(
    file_paths: Path | str | list[Path | str],
    format: Literal["VIA", "COCO"],
    images_dirs: Path | str | list[Path | str] | None = None,
) -> xr.Dataset:
    """Load an ``ethology`` bounding box annotations dataset from a file.

    Parameters
    ----------
    file_paths : pathlib.Path | str | list[pathlib.Path | str]
        Path or list of paths to the input annotation files.
    format : {"VIA", "COCO"}
        Format of the input annotation files.
    images_dirs : pathlib.Path | str | list[pathlib.Path | str], optional
        Path or list of paths to the directories containing the images the
        annotations refer to. The paths are added to the dataset
        attributes.

    Returns
    -------
    xarray.Dataset

        A valid bounding box annotations dataset with dimensions
        `image_id`, `space`, `id`, and the following arrays:

        - `position`, with dimensions (image_id, space, id),
        - `shape`, with dimensions (image_id, space, id),
        - `category`, with dimensions (image_id, id) - optional,
        - `image_shape`, with dimensions (image_id, space) - optional.

        The `category` array, if present, holds category IDs as 1-based
        integers, matching the category IDs in the input file. The dataset
        attributes include:

        - `annotation_files`: a list of paths to the input annotation files
        - `annotation_format`: the format of the input annotation files
        - `map_category_to_str`: a map from category ID to category name
        - `map_image_id_to_filename`: a map from image ID to image filename
        - `images_directories`: directory paths for the images (optional)


    Notes
    -----
    The `image_id` is assigned based on the alphabetically sorted list of
    unique image filenames across all input files. So if two images have
    the same filename but are in different input annotation files, they will
    be assigned the same image ID and their annotations will be merged.

    The `id` dimension corresponds to the annotation ID per image. It
    ranges from 0 to the maximum number of annotations per image in
    the dataset. Note that the annotation IDs are not necessarily
    consistent across images. This means that the annotations with ID=3
    in image `t` and image `t+1` will likely not correspond to the same
    individual.

    The `space` dimension holds the "x" and "y" coordinates.

    Note that supercategories are not currently added to the xarray dataset,
    even if specified in the input file.

    Examples
    --------
    Load annotations from a single COCO file:

    >>> from ethology.io.annotations import load_bboxes
    >>> ds = load_bboxes.from_files(
    ...     file_paths="path/to/annotation_file.json", format="COCO"
    ... )

    Load annotations from a single COCO file and specify the images directory:

    >>> from ethology.io.annotations import load_bboxes
    >>> ds = load_bboxes.from_files(
    ...     file_paths="path/to/annotation_file.json",
    ...     format="COCO",
    ...     images_dirs="path/to/images_dir",
    ... )

    Load annotations from two VIA files and specify multiple image directories:

    >>> from ethology.io.annotations import load_bboxes
    >>> ds = load_bboxes.from_files(
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

    # Get maps to set as dataset attributes
    map_image_id_to_filename, map_category_to_str = (
        _get_map_attributes_from_df(df_all)
    )

    # Convert dataframe to xarray dataset
    ds = _df_to_xarray_ds(df_all)

    # Add attributes to the xarray dataset
    ds.attrs = {
        "annotation_files": file_paths,
        "annotation_format": format,
        "images_directories": images_dirs,
        "map_category_to_str": map_category_to_str,
        "map_image_id_to_filename": map_image_id_to_filename,
    }

    return ds


def _get_map_attributes_from_df(
    df: DataFrame[ValidBboxAnnotationsDataFrame],
) -> tuple[dict, dict]:
    """Get the map attributes from the dataframe.

    Parameters
    ----------
    df : DataFrame[ValidBboxesDataFrame]
        Bounding box annotations dataframe.

    Returns
    -------
    tuple[dict, dict]
        Map from "image_id" to image_filename and
        map from "category_id" to category name if present.

    """
    # Compute dataset attributes from a valid intermediate dataframe
    # map from "image_id" to image_filename
    mapping_df = df[["image_filename", "image_id"]].drop_duplicates()
    map_image_id_to_filename = mapping_df.set_index("image_id").to_dict()[
        "image_filename"
    ]
    # map from "category_id" to category name
    map_category_to_str = {}
    if all(col in df.columns for col in ["category_id", "category"]):
        map_category_to_str = (
            df[["category_id", "category"]]
            .drop_duplicates()
            .set_index("category_id")
            .to_dict()["category"]
        )

        # sort by category_id
        map_category_to_str = dict(sorted(map_category_to_str.items()))

    return (map_image_id_to_filename, map_category_to_str)


@pa.check_types
def _df_from_multiple_files(
    list_filepaths: list[Path | str], format: Literal["VIA", "COCO"]
) -> DataFrame[ValidBboxAnnotationsDataFrame]:
    """Read annotations from multiple files as a valid intermediate dataframe.

    Parameters
    ----------
    list_filepaths : list[Path | str]
        List of paths to the input annotation bounding boxes files
    format : Literal["VIA", "COCO"]
        Format of the input annotation bounding boxes files.
        Currently supported formats are "VIA" and "COCO".

    Returns
    -------
    DataFrame[ValidBboxesDataFrame]
        Intermediate dataframe for bounding boxes annotations. The dataframe is
        indexed by "annotation_id" and has the following columns:
        "image_filename", "image_id", "image_width", "image_height", "x_min",
        "y_min", "width", "height", "supercategory", "category", "category_id".

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
    # NOTE: we exclude image_width and image_height from the set of columns
    # to identify duplicates, as these may differ across files.
    df_all = df_all.drop_duplicates(
        subset=[
            col
            for col in df_all.columns
            if col not in ["image_width", "image_height"]
        ],
        ignore_index=True,
        inplace=False,
    )

    # Set the index name back to "annotation_id"
    df_all.index.name = "annotation_id"

    return df_all


@pa.check_types
def _df_from_single_file(
    file_path: Path | str, format: Literal["VIA", "COCO"]
) -> DataFrame[ValidBboxAnnotationsDataFrame]:
    """Read annotations from a single file as a valid intermediate dataframe.

    Parameters
    ----------
    file_path : Path | str
        Path to the input annotation bounding boxes file.
    format : Literal["VIA", "COCO"]
        Format of the input bounding boxes annotation file.
        Currently supported formats are "VIA" and "COCO".

    Returns
    -------
    DataFrame[ValidBboxesDataFrame]
        Intermediate dataframe for bounding boxes annotations. The dataframe
        is indexed by "annotation_id" and has the following columns:
        'image_filename', 'image_id', 'image_width', 'image_height',
        'x_min', 'y_min', 'width', 'height', 'supercategory', 'category',
        'category_id'.

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

    # Cast bbox coordinates and shape as floats
    for col in ["x_min", "y_min", "width", "height"]:
        df[col] = df[col].astype(np.float64)

    # Set the index name to "annotation_id"
    df = df.set_index("annotation_id")

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

    # Get list of sorted image filenames
    image_metadata_dict = data_dict["_via_img_metadata"]
    list_sorted_filenames = sorted(
        [img_dict["filename"] for img_dict in image_metadata_dict.values()]
    )

    # Get supercategories and categories
    via_attributes = data_dict["_via_attributes"]
    supercategories_dict = {}
    if "region" in via_attributes:
        supercategories_dict = via_attributes["region"]

    # Compute list of rows in dataframe
    list_rows = []
    annotation_id = 0
    # loop through images
    for _, img_dict in image_metadata_dict.items():
        # Extract img width and height,
        # set to default if invalid or not present
        image_width, image_height = (
            _get_image_shape_attr_as_integer(
                img_dict["file_attributes"],
                file_attr,  # type: ignore
            )
            for file_attr in ["width", "height"]
        )

        # loop thru annotations in the image
        for region in img_dict["regions"]:
            # Extract region data
            region_shape = region["shape_attributes"]
            region_attributes = region["region_attributes"]

            # Extract category data if present
            if region_attributes and supercategories_dict:
                # supercategory
                # A region (bbox) can have multiple supercategories.
                # We only consider the first supercategory in alphabetical
                # order.
                supercategory = sorted(list(region_attributes.keys()))[0]

                # category name
                # in VIA files, the category_id is a string
                category_id_str = region_attributes[supercategory]
                categories_dict = supercategories_dict[supercategory][
                    "options"
                ]
                category = categories_dict[category_id_str]

                # category_id as int
                category_id = _category_id_as_int(
                    category_id_str, categories_dict
                )

            else:
                supercategory, category, category_id = (
                    ValidBboxAnnotationsDataFrame.get_empty_values()[key]
                    for key in ["supercategory", "category", "category_id"]
                )

            # Add to row
            row = {
                "annotation_id": annotation_id,
                "image_filename": img_dict["filename"],
                "image_id": list_sorted_filenames.index(img_dict["filename"]),
                "image_width": image_width,
                "image_height": image_height,
                "x_min": region_shape["x"],
                "y_min": region_shape["y"],
                "width": region_shape["width"],
                "height": region_shape["height"],
                "supercategory": supercategory,
                "category": category,
                "category_id": category_id,
            }

            list_rows.append(row)

            # update "annotation_id"
            annotation_id += 1

    return list_rows


def _get_image_shape_attr_as_integer(
    file_attrs: dict, attr_name: Literal["width", "height"]
) -> int:
    """Safely extract the image shape attribute as an integer.

    If the attribute is not present or invalid, return the default value for
    the image shape attribute defined in
    ValidBboxesDataFrame.get_empty_values().

    The file_attrs dictionary should come from a VIA input file.

    Parameters
    ----------
    file_attrs : dict
        File attributes dictionary extracted from a VIA input file.
    attr_name : Literal["width", "height"]
        Name of the image shape attribute.

    Returns
    -------
    int
        Attribute value as int. If the attribute is not present or invalid,
        return the default value for the image shape attribute defined in
        ValidBboxesDataFrame.get_empty_values().

    """
    default_value = ValidBboxAnnotationsDataFrame.get_empty_values()[
        f"image_{attr_name}"
    ]
    try:
        return int(file_attrs.get(attr_name, default_value))
    except (TypeError, ValueError):
        return default_value


def _category_id_as_int(
    category_id_str: str, list_categories: list[str]
) -> int:
    """Convert category_id to int if possible, otherwise factorize it.

    The category_id is a string in VIA files. If it cannot be converted to an
    integer, it is factorized to a 1-based integer (0 is reserved for the
    background class) based on the alphabetically sorted list of categories.

    Parameters
    ----------
    category_id_str : str
        Category ID as string.
    list_categories : list[str]
        List of categories.

    Returns
    -------
    int
        Category ID as int.

    """
    # get category_id as int
    try:
        category_id = int(category_id_str)
    except ValueError:
        # factorize to 0-based integers
        list_sorted_options = sorted(list_categories)
        category_id = list_sorted_options.index(category_id_str)
        # Add 1 to the factorised values to make them 1-based
        category_id = category_id + 1

    return category_id


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
    # We define "image_id_ethology" as the 0-based index of the image
    # following the alphabetically sorted list of unique image filenames.
    # In the following we assume the list of images under "images" in the
    # COCO JSON file is unique (i.e. it has no duplicate elements).
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
    }  # COCO files from VGG annotator always have
    # image width and height (can be 0)
    map_category_id_to_category_data = {
        cat_dict["id"]: (cat_dict["name"], cat_dict.get("supercategory", ""))
        for cat_dict in data_dict["categories"]
    }  # category data: category name, supercategory name

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
            "supercategory": supercategory,  # if not defined, set to ""
            "category": category,
            "category_id": category_id,
            # in COCO files, the category_id is always a 1-based integer
        }

        list_rows.append(row)

    return list_rows


@pa.check_types
def _df_to_xarray_ds(
    df: DataFrame[ValidBboxAnnotationsDataFrame],
) -> xr.Dataset:
    """Convert a bounding box annotations dataframe to an xarray dataset.

    Parameters
    ----------
    df : DataFrame[ValidBboxesDataFrame]
        A valid intermediate dataframe for bounding boxes annotations.

    Returns
    -------
    xr.Dataset
        an xarray dataset with the following dimensions:
        - `image_id`: holds the 0-based index of the image in the "images"
        list of the COCO JSON file;
        - `space`: `x` or `y`;
        - `id`: annotation ID per image, assigned from 0 to the max number of
        annotations per image in the full dataset. Note that the annotation IDs
        are not necessarily consistent across images. This means that the
        annotations with ID `m` in image `t` and image `t+1` will likely not
        correspond to the same individual.

        The dataset is made up of the following arrays:
        - `position`: (`image_id`, `space`, `id`)
        - `shape`: (`image_id`, `space`, `id`)
        - `category`: (`image_id`, `id`)

    """
    # Drop columns if all values in that column are empty
    default_values = ValidBboxAnnotationsDataFrame.get_empty_values()
    list_empty_cols = [
        col for col in default_values if all(df[col] == default_values[col])
    ]
    df = df.drop(columns=list_empty_cols)

    # Compute max number of annotations per image
    max_annotations_per_image = df["image_id"].value_counts().max()

    # Sort the dataframe by image_id
    df = df.sort_values(by=["image_id"])

    # Compute indices of the rows where the image ID switches
    bool_id_diff_from_prev = df["image_id"].ne(df["image_id"].shift())
    indices_id_switch = np.argwhere(bool_id_diff_from_prev)[1:, 0]

    # Extract arrays from the dataframe
    arrays_metadata = _prepare_array_dicts(df)
    array_dict = _extract_arrays_from_df(
        df, arrays_metadata, indices_id_switch, max_annotations_per_image
    )

    # Build data vars dictionary
    data_vars = {
        array_key.split("_array")[0]: (
            arrays_metadata[array_key]["dims"],
            array_dict[array_key],
        )
        for array_key in array_dict
    }

    return xr.Dataset(
        data_vars=data_vars,
        coords=dict(
            image_id=df["image_id"].unique(),
            space=["x", "y"],
            id=range(max_annotations_per_image),
        ),
    )


def _prepare_array_dicts(
    df: pd.DataFrame,
) -> dict[str, dict[str, Any]]:
    """Prepare the metadata for the arrays in the xarray dataset.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe for bounding boxes annotations.

    Returns
    -------
    dict[str, dict[str, Any]]
        A dictionary with the metadata for the arrays in the xarray dataset.

    """
    arrays_metadata: dict[str, dict[str, Any]] = {
        "position_array": {
            "columns": ["x_min", "y_min"],
            "type": np.float64,
            "pad_value": np.nan,
            "dims": ("image_id", "space", "id"),
        },
        "shape_array": {
            "columns": ["width", "height"],
            "type": np.float64,
            "pad_value": np.nan,
            "dims": ("image_id", "space", "id"),
        },
    }

    # Add image shape data if present
    if all(col in df.columns for col in ["image_width", "image_height"]):
        arrays_metadata["image_shape_array"] = {
            "columns": ["image_width", "image_height"],
            "type": int,
            "pad_value": -1,
            "dims": ("image_id", "space"),
        }

    # Add category data if present
    if all(col in df.columns for col in ["category_id", "category"]):
        arrays_metadata["category_array"] = {
            "columns": ["category_id"],
            "type": int,
            "pad_value": -1,
            "dims": ("image_id", "id"),
        }

    return arrays_metadata


def _extract_arrays_from_df(
    df: pd.DataFrame,
    arrays_metadata: dict[str, dict[str, Any]],
    indices_id_switch: np.ndarray,
    max_annotations_per_image: int,
) -> dict[str, np.ndarray]:
    """Extract arrays in metadata dict from a df of bounding boxes annotations.

    Parameters
    ----------
    df : pd.DataFrame
        A dataframe for bounding boxes annotations.
    arrays_metadata : dict[str, dict[str, Any]]
        A dictionary with the metadata for the arrays to extract.
    indices_id_switch : np.ndarray
        Indices of the rows where the image ID switches.
    max_annotations_per_image : int
        The maximum number of annotations per image.

    Returns
    -------
    dict[str, np.ndarray]
        A dictionary with the arrays extracted from the dataframe.

    """
    array_dict = {}
    for key in arrays_metadata:
        # Extract annotations per image
        list_arrays = np.split(
            df[arrays_metadata[key]["columns"]].to_numpy(
                dtype=arrays_metadata[key]["type"]
            ),
            indices_id_switch,
        )  # each array: (n_annotations, N_DIM)

        if key == "image_shape_array":
            array_dict[key] = np.stack(
                [np.unique(arr, axis=0) for arr in list_arrays], axis=0
            ).squeeze(axis=1)  # (n_images, N_DIM)

        else:
            # Pad arrays with NaN values along the annotation ID axis
            # and stack to (n_images, n_max_annotations, N_DIM)
            list_arrays_padded = [
                np.pad(
                    arr,
                    ((0, max_annotations_per_image - arr.shape[0]), (0, 0)),
                    constant_values=arrays_metadata[key]["pad_value"],
                )
                for arr in list_arrays
            ]
            array_dict[key] = np.stack(list_arrays_padded, axis=0)

            # Reorder dimensions to (n_images, N_DIM, n_max_annotations)
            # (squeeze the N_DIM axis (N_DIM=1) for "category")
            array_dict[key] = np.moveaxis(array_dict[key], -1, 1)
            if key == "category_array":
                array_dict[key] = array_dict[key].squeeze(axis=1)

    # Modify x_min and y_min to represent the bbox centre
    array_dict["position_array"] += array_dict["shape_array"] / 2

    return array_dict
