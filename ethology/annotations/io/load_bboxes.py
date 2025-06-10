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


# --- NEW FUNCTION (for issue #43) ---
def _detect_format(file_path: Path) -> Literal["VIA", "COCO"]:
    """Detect the format (VIA or COCO) of a JSON annotation file.

    Detection is based on the presence of characteristic top-level keys.

    Parameters
    ----------
    file_path : Path
        Path to the input annotation file.

    Returns
    -------
    Literal["VIA", "COCO"]
        The detected format.

    Raises
    ------
    ValueError
        If the file cannot be read or parsed, or if the format cannot
        be reliably determined from the top-level keys.

    """
    try:
        with open(file_path) as f:
            data = json.load(f)
    except Exception as e:
        # Keep this as a fallback for truly weird read issues
        raise ValueError(
            f"Could not read or parse JSON from file {file_path} "
            f"for format detection: {e}"
        ) from e

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected JSON root to be a dictionary for format detection, "
            f"but got {type(data)} in file {file_path}"
        )

    # The validators will check if data is a dict,
    # so we don't need to check here
    top_level_keys = set(data.keys())

    # Define characteristic keys
    via_keys = {"_via_img_metadata", "_via_attributes", "_via_settings"}
    coco_keys = {"images", "annotations", "categories"}

    has_via_keys = bool(via_keys.intersection(top_level_keys))
    has_coco_keys = coco_keys.issubset(top_level_keys)

    if has_coco_keys and has_via_keys:
        raise ValueError(
            f"File {file_path} contains keys characteristic of *both* VIA "
            "and COCO formats. "
            "Cannot reliably determine format."
        )
    elif has_coco_keys:
        return "COCO"
    elif has_via_keys:
        return "VIA"
    else:
        raise ValueError(
            f"Could not automatically determine format for file {file_path}. "
            "File does not contain characteristic top-level keys for VIA or "
            "COCO."
        )


def _determine_format_from_paths(
    input_file_list: list[Path],
) -> Literal["VIA", "COCO"]:
    """Determine the annotation format by inspecting files.

    If multiple files are provided, ensure they are all of the same detected
    format.
    """
    try:
        # Should be caught by from_files, but good check
        if not input_file_list:
            raise ValueError(
                "Cannot determine format from an empty list of files."
            )

        if len(input_file_list) == 1:
            # Detect format based on the single file
            return _detect_format(input_file_list[0])
        else:
            # For multiple files, check all files and ensure consistency
            detected_formats = []
            for file_path in input_file_list:
                detected_format = _detect_format(file_path)
                detected_formats.append(detected_format)

            # Check if all formats are the same
            unique_formats = set(detected_formats)
            if len(unique_formats) > 1:
                raise ValueError(
                    f"Inconsistent formats detected across files: "
                    f"{unique_formats}. All files must have the same format."
                )
            return detected_formats[0]
    except (
        FileNotFoundError,
        ValueError,
    ) as e:  # Catch errors from _detect_format
        # Re-raise errors related to detection more informatively
        raise ValueError(f"Automatic format detection failed: {e}") from e


# --- REFACTORED from_files FUNCTION ---
def from_files(
    file_paths: Path | str | list[Path | str],
    format: Literal["VIA", "COCO", "auto"] = "auto",
    images_dirs: Path | str | list[Path | str] | None = None,
) -> pd.DataFrame:
    """Read input annotation files as a bboxes dataframe.

    Parameters
    ----------
    file_paths : Path | str | list[Path | str]
        Path or list of paths to the input annotation files.
    format : Literal["VIA", "COCO", "auto"], optional
        Format of the input annotation files. If set to "auto" (default),
        the format will be detected based on the content of the files
        provided. Detection relies on characteristic top-level keys in the
        JSON structure. For multiple files, all files must have the same
        format.
    images_dirs : Path | str | list[Path | str], optional
        Path or list of paths to the directories containing the images the
        annotations refer to.

    Returns
    -------
    pd.DataFrame
        Bounding boxes annotations dataframe. The dataframe is indexed
        by "annotation_id" and has the standard columns (see Notes).
        It also has the following attributes: "annotation_files",
        "annotation_format", "images_directories".

    Raises
    ------
    ValueError
        If format="auto" and the format cannot be detected, if multiple
        files have inconsistent formats, or if an invalid format string
        is provided.
    FileNotFoundError
        If format="auto" and any file path does not exist (this will be
        the underlying cause of the ValueError from auto-detection).
    json.JSONDecodeError
        If format="auto" and any file cannot be parsed as JSON (this
        will be the underlying cause of the ValueError from auto-detection).
    TypeError
        If `file_paths` is of an unsupported type.

    Notes
    -----
    The standard dataframe has the following columns: "image_filename",
    "image_id", "image_width", "image_height", "x_min", "y_min",
    "width", "height", "supercategory", "category", "category_id".

    The "image_id" is assigned based on the alphabetically sorted list of
    unique image filenames across all input files. The "category_id" column
    is always a 0-based integer derived from the category names.

    When loading multiple files:

    - If `format="auto"`, the format is detected from all files and they must
      all have the same format.
    - Image filenames are used to assign unique image IDs. If the same
      filename appears in multiple annotation files, annotations will be
      merged under the same `image_id`.
    - Duplicate annotations across files are dropped.

    See Also
    --------
    pandas.concat : Concatenate pandas objects along a particular axis.
    pandas.DataFrame.drop_duplicates : Return DataFrame with duplicate rows
        removed.

    """
    # Ensure file_paths is a list internally, even if single path is given
    if isinstance(file_paths, str | Path):
        input_file_list = [Path(file_paths)]
        is_single_file = True
    elif isinstance(file_paths, list):
        if not file_paths:
            raise ValueError("Input 'file_paths' list cannot be empty.")
        input_file_list = [Path(p) for p in file_paths]
        is_single_file = False
    else:
        raise TypeError(
            f"Unsupported type for 'file_paths': {type(file_paths)}"
        )

    # --- Determine Format ---
    determined_format: Literal["VIA", "COCO"]
    if format == "auto":
        determined_format = _determine_format_from_paths(input_file_list)
    elif format in ["VIA", "COCO"]:
        determined_format = format
    else:
        raise ValueError(
            f"Invalid format specified: '{format}'. Must be 'VIA', "
            f"'COCO', or 'auto'."
        )
    # --- End Determine Format ---

    # Delegate to reader of either a single file or multiple files
    if is_single_file:  # or len(input_file_list) == 1
        df_all = _from_single_file(
            input_file_list[0], format=determined_format
        )
    else:
        # Pass the list of Path objects
        df_all = _from_multiple_files(
            input_file_list, format=determined_format
        )

    # Add metadata
    df_all.attrs = {
        "annotation_files": file_paths,  # Store original input representation
        "annotation_format": determined_format,
        "images_directories": images_dirs,
    }

    return df_all


def _from_multiple_files(
    list_filepaths: list[Path], format: Literal["VIA", "COCO"]
):
    """Read bounding boxes annotations from multiple files.

    Parameters
    ----------
    list_filepaths : list[Path]
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
    elif format == "COCO":
        # In COCO files exported with the VIA tool, the category_id
        # is always a 1-based integer. Here we coerce it to a 0-based
        # integer
        df["category_id"] = df["category"].factorize(sort=True)[0]

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
