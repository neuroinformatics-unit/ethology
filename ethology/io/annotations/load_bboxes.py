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
    *,
    retain_image_id: bool = False,
) -> xr.Dataset:
    """Load an ``ethology`` bounding box annotations dataset.

    Parameters
    ----------
    file_paths
        Path or list of paths to the input annotation files.
    format
        Format of the input annotation files.
    images_dirs
        Paths to the directories containing the images the annotations
        refer to. The paths are added to dataset attributes.
    retain_image_id
        If True and supported by the input format, preserve the image IDs
        as they appear in the input file (e.g., COCO images[].id).
        If False (default) keep ethology's behaviour of renumbering
        images as 0-based indices sorted by filename.

    """
    # Optionally build filename -> original id map (COCO or VIA)
    filename_to_original_id: dict[str, int] = {}
    if retain_image_id:
        list_files = (
            list(file_paths) if isinstance(file_paths, list) else [file_paths]
        )
        filename_to_original_id = _compute_filename_to_original_id(
            list_files, format
        )

    # Load annotations into the intermediate dataframe using helpers.
    if isinstance(file_paths, list):
        df_all = _df_from_multiple_files(list(file_paths), format=format)
    else:
        df_all = _df_from_single_file(file_paths, format=format)

    # If requested, apply original IDs where available and get mapping:
    # ethology_image_id -> original_image_id.
    map_image_id_to_original: dict[int, int] = {}
    if retain_image_id and filename_to_original_id:
        df_all, map_image_id_to_original = _apply_original_ids(
            df_all, filename_to_original_id
        )

    # Build attribute maps and convert to xarray dataset.
    map_image_id_to_filename, map_category_to_str = (
        _get_map_attributes_from_df(df_all)
    )

    ds = _df_to_xarray_ds(df_all)
    ds.attrs = {
        "annotation_files": file_paths,
        "annotation_format": format,
        "images_directories": images_dirs,
        "map_category_to_str": map_category_to_str,
        "map_image_id_to_filename": map_image_id_to_filename,
        "map_image_id_to_original": map_image_id_to_original,
    }

    return ds


def _compute_filename_to_original_id(
    list_files: list[Path | str], format: Literal["VIA", "COCO"]
) -> dict[str, int]:
    """Dispatch to the format-specific filename->original-id builder."""
    if format == "COCO":
        return _compute_filename_to_original_id_coco(list_files)
    if format == "VIA":
        return _compute_filename_to_original_id_via(list_files)
    return {}


def _compute_filename_to_original_id_coco(
    list_files: list[Path | str],
) -> dict[str, int]:
    """Build filename -> COCO image id mapping from COCO files."""
    mapping: dict[str, int] = {}
    for fp in list_files:
        p = Path(fp)
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            # not JSON or unreadable; skip
            continue
        for img in data.get("images", []):
            fname = img.get("file_name")
            if fname is not None:
                # prefer the last-seen mapping across files
                mapping[fname] = img.get("id")
    return mapping


def _compute_filename_to_original_id_via(
    list_files: list[Path | str],
) -> dict[str, int]:
    """Build filename -> VIA metadata-key-as-int mapping where possible."""
    mapping: dict[str, int] = {}
    for fp in list_files:
        p = Path(fp)
        if not p.exists():
            continue
        try:
            with p.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except Exception:
            continue
        md = data.get("_via_img_metadata", {})
        for img_key, img_dict in md.items():
            fname = img_dict.get("filename")
            try:
                orig_id = int(img_key)
            except Exception:
                orig_id = None
            if fname is not None and orig_id is not None:
                mapping[fname] = orig_id
    return mapping


def _apply_original_ids(
    df_all: pd.DataFrame, filename_to_original_id: dict[str, int]
) -> tuple[pd.DataFrame, dict[int, int]]:
    """Apply filename->original-id mapping to dataframe.

    Return updated dataframe and map ethology_image_id -> original id.
    """
    # Build a small dataframe mapping ethology image_id -> filename
    mapping_df = (
        df_all[["image_filename", "image_id"]].drop_duplicates().copy()
    )

    # Ensure image_id column is plain Python int values (for indexing).
    mapping_df["image_id"] = mapping_df["image_id"].astype(int)

    # Map filename -> original id (may yield NaN where no mapping exists).
    mapping_df["image_id_original"] = mapping_df["image_filename"].map(
        filename_to_original_id
    )

    # Keep only rows where an original id exists and produce a dict
    # with plain Python ints for both keys and values.
    mapping_series = (
        mapping_df.dropna(subset=["image_id_original"])
        .set_index("image_id")["image_id_original"]
        .astype(int)
    )
    raw_map: dict[Any, int] = mapping_series.to_dict()
    map_image_id_to_original: dict[int, int] = {
        int(k): int(v) for k, v in raw_map.items()
    }

    # Overwrite df_all["image_id"] where mapping exists; otherwise keep
    # ethology-assigned id.
    df_all["image_id"] = (
        df_all["image_filename"]
        .map(filename_to_original_id)
        .fillna(df_all["image_id"])
        .astype(int)
    )

    return df_all, map_image_id_to_original


def _get_map_attributes_from_df(
    df: DataFrame[ValidBboxAnnotationsDataFrame],
) -> tuple[dict[int, str], dict[int, str]]:
    """Get dataset attribute maps (img_id->filename, cat_id->name)."""
    mapping_df = df[["image_filename", "image_id"]].drop_duplicates()
    map_image_id_to_filename = mapping_df.set_index("image_id").to_dict()[
        "image_filename"
    ]

    map_category_to_str: dict[int, str] = {}
    if all(col in df.columns for col in ["category_id", "category"]):
        map_category_to_str = (
            df[["category_id", "category"]]
            .drop_duplicates()
            .set_index("category_id")
            .to_dict()["category"]
        )
        map_category_to_str = dict(sorted(map_category_to_str.items()))

    return (map_image_id_to_filename, map_category_to_str)


@pa.check_types
def _df_from_multiple_files(
    list_filepaths: list[Path | str], format: Literal["VIA", "COCO"]
) -> DataFrame[ValidBboxAnnotationsDataFrame]:
    """Read annotations from multiple files as a dataframe."""
    df_list = [
        _df_from_single_file(file, format=format) for file in list_filepaths
    ]

    df_all = pd.concat(df_list, ignore_index=True)

    list_image_filenames = sorted(list(df_all["image_filename"].unique()))
    df_all["image_id"] = df_all["image_filename"].apply(
        lambda x: list_image_filenames.index(x)
    )

    df_all = df_all.sort_values(by=["image_filename"])

    df_all = df_all.drop_duplicates(
        subset=[
            col
            for col in df_all.columns
            if col not in ["image_width", "image_height"]
        ],
        ignore_index=True,
        inplace=False,
    )

    df_all.index.name = "annotation_id"

    return df_all


@pa.check_types
def _df_from_single_file(
    file_path: Path | str, format: Literal["VIA", "COCO"]
) -> DataFrame[ValidBboxAnnotationsDataFrame]:
    """Read annotations from a single file as a dataframe."""
    validator: type[ValidVIA | ValidCOCO]
    if format == "VIA":
        validator = ValidVIA
        get_rows_from_file = _df_rows_from_valid_VIA_file
    elif format == "COCO":
        validator = ValidCOCO
        get_rows_from_file = _df_rows_from_valid_COCO_file
    else:
        raise ValueError(f"Unsupported format: {format}")

    valid_file = validator(file_path)
    list_rows = get_rows_from_file(valid_file.path)
    df = pd.DataFrame(list_rows)

    df = df.sort_values(by=["image_filename"])

    df = df.drop_duplicates(
        subset=[col for col in df.columns if col != "annotation_id"],
        ignore_index=True,
        inplace=False,
    )

    for col in ["x_min", "y_min", "width", "height"]:
        df[col] = df[col].astype(np.float64)

    df = df.set_index("annotation_id")

    return df


def _df_rows_from_valid_VIA_file(file_path: Path) -> list[dict]:
    """Extract rows from a validated VIA JSON file."""
    with open(file_path) as file:
        data_dict = json.load(file)

    image_metadata_dict = data_dict["_via_img_metadata"]
    list_sorted_filenames = sorted(
        [img_dict["filename"] for img_dict in image_metadata_dict.values()]
    )

    via_attributes = data_dict["_via_attributes"]
    supercategories_dict = {}
    if "region" in via_attributes:
        supercategories_dict = via_attributes["region"]

    list_rows: list[dict] = []
    annotation_id = 0
    for _, img_dict in image_metadata_dict.items():
        image_width = _get_image_shape_attr_as_integer(
            img_dict["file_attributes"], "width"
        )
        image_height = _get_image_shape_attr_as_integer(
            img_dict["file_attributes"], "height"
        )

        for region in img_dict["regions"]:
            region_shape = region["shape_attributes"]
            region_attributes = region["region_attributes"]

            if region_attributes and supercategories_dict:
                supercategory = sorted(list(region_attributes.keys()))[0]
                category_id_str = region_attributes[supercategory]
                categories_dict = supercategories_dict[supercategory][
                    "options"
                ]
                category = categories_dict[category_id_str]
                category_id = _category_id_as_int(
                    category_id_str, categories_dict
                )
            else:
                supercategory, category, category_id = (
                    ValidBboxAnnotationsDataFrame.get_empty_values()[key]
                    for key in ["supercategory", "category", "category_id"]
                )

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
            annotation_id += 1

    return list_rows


def _get_image_shape_attr_as_integer(
    file_attrs: dict, attr_name: Literal["width", "height"]
) -> int:
    """Safely extract the image shape attribute as an integer."""
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
    """Convert category_id to int if possible, otherwise factorize it."""
    try:
        category_id = int(category_id_str)
    except ValueError:
        list_sorted_options = sorted(list_categories)
        category_id = list_sorted_options.index(category_id_str)
        category_id = category_id + 1
    return category_id


def _df_rows_from_valid_COCO_file(file_path: Path) -> list[dict]:
    """Extract rows from a validated COCO JSON file."""
    with open(file_path) as file:
        data_dict = json.load(file)

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
        cat_dict["id"]: (cat_dict["name"], cat_dict.get("supercategory", ""))
        for cat_dict in data_dict["categories"]
    }

    list_rows: list[dict] = []
    for annot_id, annot_dict in enumerate(data_dict["annotations"]):
        img_id_coco = annot_dict["image_id"]
        image_filename = map_img_id_coco_to_filename[img_id_coco]
        image_width, image_height = map_img_id_coco_to_width_height[
            img_id_coco
        ]
        img_id_ethology = map_img_id_coco_to_ethology[img_id_coco]

        x_min, y_min, width, height = annot_dict["bbox"]
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
        }
        list_rows.append(row)

    return list_rows


@pa.check_types
def _df_to_xarray_ds(
    df: DataFrame[ValidBboxAnnotationsDataFrame],
) -> xr.Dataset:
    """Convert bounding box dataframe to an xarray dataset."""
    default_values = ValidBboxAnnotationsDataFrame.get_empty_values()
    list_empty_cols = [
        col for col in default_values if all(df[col] == default_values[col])
    ]
    df = df.drop(columns=list_empty_cols)

    max_annotations_per_image = df["image_id"].value_counts().max()
    df = df.sort_values(by=["image_id"])

    bool_id_diff_from_prev = df["image_id"].ne(df["image_id"].shift())
    indices_id_switch = np.argwhere(bool_id_diff_from_prev)[1:, 0]

    arrays_metadata = _prepare_array_dicts(df)
    array_dict = _extract_arrays_from_df(
        df, arrays_metadata, indices_id_switch, max_annotations_per_image
    )

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
    """Prepare the metadata for arrays in the xarray dataset."""
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

    if all(col in df.columns for col in ["image_width", "image_height"]):
        arrays_metadata["image_shape_array"] = {
            "columns": ["image_width", "image_height"],
            "type": int,
            "pad_value": -1,
            "dims": ("image_id", "space"),
        }

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
    """Extract arrays from dataframe using arrays_metadata."""
    array_dict: dict[str, np.ndarray] = {}
    for key in arrays_metadata:
        list_arrays = np.split(
            df[arrays_metadata[key]["columns"]].to_numpy(
                dtype=arrays_metadata[key]["type"]
            ),
            indices_id_switch,
        )

        if key == "image_shape_array":
            array_dict[key] = np.stack(
                [np.unique(arr, axis=0) for arr in list_arrays], axis=0
            ).squeeze(axis=1)
        else:
            list_arrays_padded = [
                np.pad(
                    arr,
                    ((0, max_annotations_per_image - arr.shape[0]), (0, 0)),
                    constant_values=arrays_metadata[key]["pad_value"],
                )
                for arr in list_arrays
            ]
            array_dict[key] = np.stack(list_arrays_padded, axis=0)
            array_dict[key] = np.moveaxis(array_dict[key], -1, 1)
            if key == "category_array":
                array_dict[key] = array_dict[key].squeeze(axis=1)

    array_dict["position_array"] += array_dict["shape_array"] / 2

    return array_dict
